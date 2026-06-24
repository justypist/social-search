# 视频画面文字提取流水线

本文是视觉文字提取功能的实现设计，不是当前已上线能力。当前仓库只支持字幕下载和音频转写；视觉提取必须由用户在创建任务时显式勾选后才运行。

## 触发条件

视觉提取是任务级选项，默认关闭。

- 用户创建任务时勾选“提取视频画面文字”，前端提交 `extract_visual: true`。
- 未勾选时保持现有行为：不下载视频用于视觉分析，不抽帧，不加载 OCR，不生成 `pages.json`。
- 已勾选时，即使命中了视频自带字幕，也必须下载视频并运行视觉流水线。

任务创建链路需要新增同一个字段：

```python
class CreateTaskRequest(BaseModel):
    url: str = Field(min_length=1, max_length=4096)
    language: Literal["auto", "zh", "en"] | None = None
    extract_visual: bool = False
```

字段需要继续传递到：

- 前端 `taskForm`：新增 checkbox，提交体包含 `{ url, language, extract_visual }`
- `TaskRecord`：持久化用户创建任务时的选择
- `TaskManager._job_payload()`：写入 worker job JSON
- `api.worker._build_config()`：构造 `ExtractConfig`
- `ExtractConfig`：新增 `extract_visual: bool = False`

如果视觉提取失败，任务应默认失败，而不是静默产出只有字幕的结果。这样可以避免用户误以为画面文字已经进入索引。如果后续需要宽松模式，再单独增加 `visual_optional`。

## 目标场景

目标是从视频画面中提取可搜索文字，主要覆盖文档讲解类视频：

- PPT、PDF、Word、论文、网页、代码截图等带有规整文字的画面
- 翻页、滚动、分步动画出现的文字增量
- 字幕以外、语音转写无法覆盖的画面文字

非目标场景包括纯真人讲解、户外运动、游戏画面、无稳定文字的信息流。这些画面应尽早过滤，避免 OCR 空消耗。

## 当前流程变化

现有提取流程是：

```text
probe -> prepare -> 下载字幕或音频转写 -> 写 subtitle/transcript/meta
```

新增视觉选项后的流程是：

```text
probe
 ↓
prepare
 ↓
尝试下载字幕
 ↓
如果 extract_visual=True，确保有 video 文件
 ↓
如果没有字幕：
  - extract_visual=True：优先从已下载 video 抽音频后 Whisper 转写
  - extract_visual=False：保持现有 audio-first，失败后 video fallback
 ↓
如果 extract_visual=True，运行视觉文字提取
 ↓
写 subtitle/transcript/pages/meta
```

关键点：

- `extract_visual=False` 时现有行为不变。
- `extract_visual=True` 时视频下载不是音频失败兜底，而是视觉流水线的必要输入。
- 如果已因视觉提取下载了视频，后续 Whisper 转写应复用同一个视频抽音频，避免同时下载 audio 和 video。
- 如果 `keep_media=False`，只清理下载的 `audio.*` / `video.*`；`pages.json` 和被页面引用的代表帧属于输出产物，应保留。

## 整体流水线

```text
视频文件
 ↓
1. 抽帧，默认 1fps
 ↓
2. has_text 轻量判定
   ├─ 否：标记为 non_text，不进入 OCR
   └─ 是：进入文字帧段
 ↓
3. 帧差检测，仅在文字帧段内
   SSIM 或 pHash 超过阈值 -> 候选页面变化
 ↓
4. 稳定检测和代表帧选取
   候选变化后等待画面稳定，取稳定区间代表帧
 ↓
5. OCR
   只对代表帧识别文字
 ↓
6. 文字密度过滤和去重
   文本过短丢弃，相邻页高度相似或包含关系则合并
 ↓
7. 输出 pages.json，并在 meta.json.files 登记
```

四层过滤分工：

| 层 | 目的 |
|---|---|
| `has_text` | 过滤明显没有可索引文字的画面，减少 OCR 成本 |
| 帧差阈值 | 过滤鼠标滑动、激光笔、小范围指针移动 |
| 稳定检测 | 避免取到翻页动画、滚动中间态、淡入过渡帧 |
| OCR 去重 | 兜底处理动画分步、重复页、误判候选变化 |

## 抽帧

默认 1fps，先满足文档翻页场景。后续可按任务配置调高，例如代码滚动或快速翻页视频。

```bash
ffmpeg -i video.mp4 -vf "fps=1" -q:v 2 frames/%06d.jpg
```

这个命令只生成图片，不会自动生成时间戳索引。`FrameExtractor` 需要同时写 `frames.json`：

```json
{
  "fps": 1.0,
  "frames": [
    { "index": 0, "timestamp": 0.0, "path": "frames/000001.jpg" },
    { "index": 1, "timestamp": 1.0, "path": "frames/000002.jpg" }
  ]
}
```

若后续需要更精确的时间戳，可改用 `ffprobe` 或 ffmpeg `showinfo` 解析真实 PTS。

## has_text 判定

`has_text` 是无模型成本的负向过滤器，只负责排除明显没有文字的画面，不能做过强假设。不要因为画面方差高就直接判无文字，因为代码截图、论文页、网页和产品画面都可能有高方差。

初始实现建议使用 OpenCV 特征：

```python
def has_text(frame: np.ndarray, config: ExtractConfig) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 纯黑、纯白、过渡帧通常没有可索引文字。
    if gray.var() < config.has_text_variance_min:
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin = cv2.inRange(hsv, (0, 30, 60), (20, 150, 255))
    skin_ratio = float((skin > 0).mean())
    if skin_ratio > config.has_text_skin_ratio_max:
        return False

    edges = cv2.Canny(gray, 50, 150)
    row_profile = edges.mean(axis=1)
    if row_profile.std() < config.has_text_row_edge_std_min:
        return False

    return True
```

注意点：

- `cv2.inRange()` 的输出是 `0/255` 掩码，肤色占比必须用 `(skin > 0).mean()` 或 `skin.mean() / 255`。
- `has_text` 只作为第一层粗过滤，阈值必须用实际视频样本校准。
- 需要连续 `N` 帧为 `True` 才开始文字段，避免切换瞬间的单帧抖动。
- 非文字段写入 `pages.json.non_text_segments`，便于后续确认哪些时间段没有视觉文字。

## 页面变化检测

页面变化检测只在连续文字帧段内运行。推荐先实现 SSIM，再视性能需求补 pHash。

```python
from skimage.metrics import structural_similarity as ssim

score = ssim(prev_gray, curr_gray)
if score < config.page_change_ssim_threshold:
    # 候选页面变化
```

pHash 可作为轻量替代或辅助：

```python
import imagehash
from PIL import Image

prev_hash = imagehash.phash(Image.fromarray(prev_frame))
curr_hash = imagehash.phash(Image.fromarray(curr_frame))
dist = prev_hash - curr_hash
if dist > config.page_change_phash_threshold:
    # 候选页面变化
```

阈值不要写死为绝对正确值。建议从以下默认开始，用样本校准：

```python
page_change_ssim_threshold: float = 0.85
page_change_phash_threshold: int = 20
```

对持续小动画的视频，可增加滑动窗口基线：

```python
baseline = moving_average(hamming_dists, window=30)
is_page_turn = (dist - baseline) > config.page_change_phash_delta
```

## 稳定检测和代表帧

候选页面变化后，不立刻 OCR 当前帧，而是等待画面稳定：

```python
# 候选变化后，连续 stable_frame_count 帧 SSIM > stable_ssim_threshold
# 判定为稳定区间，取稳定区间中间帧或第一张清晰帧作为代表帧。
stable_frame_count: int = 2
stable_ssim_threshold: float = 0.95
```

如果视频只有 1fps，`stable_frame_count=2` 意味着需要约 2 秒确认稳定；这对文档讲解通常可接受。快速翻页样本需要调高 `frame_fps` 或降低稳定帧要求。

## OCR

推荐使用当前 RapidOCR 主包，而不是旧的拆分包。

官方文档截至 2026-06-24 的要点：

- `rapidocr_onnxruntime`、`rapidocr_openvino`、`rapidocr_paddle` 三个库逐渐不再维护，后续以 `rapidocr` 为主。
- `rapidocr>=2.0.6` 不再把 ONNX Runtime 作为依赖包，需要手动安装推理引擎。
- 官方示例推荐先使用 ONNX Runtime CPU 版：`pip install rapidocr onnxruntime`。
- GPU 推理需要安装和配置对应推理引擎，不能在本项目里默认承诺可用。

示例：

```python
from rapidocr import RapidOCR

ocr = RapidOCR()
result = ocr(frame_path)

texts = list(result.txts or ())
boxes = result.boxes
scores = list(result.scores or ())
```

本项目当前 `pyproject.toml` 要求 Python `>=3.14`，RapidOCR 依赖需要在这个运行环境里实测安装。如果依赖暂不支持 Python 3.14，可选方案是降低项目 Python 版本约束，或把 OCR 放到独立 Python 环境/子进程中。

## 文字过滤和去重

OCR 后做最后一层过滤：

```python
if len(normalized_text) < config.visual_text_min_chars:
    discard_page()
```

相邻页去重不能只依赖 Jaccard。分步动画常见模式是“当前页包含上一页内容，并新增一个 bullet”，Jaccard 可能低于 0.9，但它仍然应该合并为同一页的更新版本。

建议合并条件：

```python
similar = jaccard_similarity(prev_text, curr_text) > config.text_dedup_jaccard_threshold
contained = containment_ratio(shorter_text, longer_text) > config.text_dedup_containment_threshold

if similar or contained:
    merge_pages(prev_page, curr_page)
```

合并策略：

- `start` 保留上一页开始时间
- `end` 更新为当前页结束时间
- `text` 取信息量更多的一版
- `frame_path` 取 OCR 置信度更高或文字更多的代表帧

## 输出格式

`pages.json` 与现有 `Transcript.segments` 的时间轴对齐，便于后续合并索引。

```json
{
  "frame_fps": 1.0,
  "pages": [
    {
      "page_index": 0,
      "start": 12.0,
      "end": 45.0,
      "text": "标题\n正文第一行\n正文第二行",
      "frame_path": "frames/page_0000.jpg",
      "confidence": 0.93
    }
  ],
  "non_text_segments": [
    { "start": 0.0, "end": 12.0, "reason": "no_text" }
  ],
  "stats": {
    "sampled_frames": 120,
    "text_frames": 64,
    "ocr_frames": 18,
    "pages": 10
  }
}
```

`meta.json.files` 在开启视觉提取时增加：

```json
{
  "pages_json": "pages.json",
  "visual_frames": "frames"
}
```

`ExtractionResult` 增加可空字段：

```python
pages_json_path: Path | None = None
frames_dir: Path | None = None
```

## 新增模块

沿用当前项目风格：Protocol、dataclass、进度回调、文件输出。

- `frames.py`：`FrameExtractor` Protocol 和 ffmpeg 实现，负责抽帧和写 `frames.json`
- `ocr.py`：`OcrRecognizer` Protocol 和 RapidOCR 实现
- `vision.py`：`HasTextDetector`、`PageChangeDetector`、`StabilityDetector`、`TextDedup` 等纯逻辑组件
- `visual.py`：编排视觉流水线，输入 `video_path`，输出 `pages.json`

`ExtractConfig` 建议新增：

```python
extract_visual: bool = False
frame_fps: float = 1.0
has_text_variance_min: float = 20.0
has_text_skin_ratio_max: float = 0.15
has_text_row_edge_std_min: float = 1.5
page_change_ssim_threshold: float = 0.85
page_change_phash_threshold: int = 20
page_change_phash_delta: int = 12
stable_frame_count: int = 2
stable_ssim_threshold: float = 0.95
visual_text_min_chars: int = 10
text_dedup_jaccard_threshold: float = 0.9
text_dedup_containment_threshold: float = 0.85
```

依赖建议：

```toml
dependencies = [
    "opencv-python-headless",
    "scikit-image",
    "rapidocr",
    "onnxruntime",
]
```

如果启用 pHash，再加入：

```toml
dependencies = [
    "imagehash",
    "Pillow",
]
```

## 进度阶段

前端 `STAGE_LABELS` 需要补充视觉阶段：

```javascript
visual_prepare: "准备视觉提取",
visual_frames: "抽取视频帧",
visual_detect: "检测文字画面",
visual_ocr: "识别画面文字",
visual_write: "写入画面文字",
```

整体进度要根据 `extract_visual` 动态分配。未勾选时保持当前进度区间；勾选后把视频下载、抽帧、OCR 纳入总进度，避免任务长时间卡在“写入文件”或“完成前”。

## 测试要点

最低测试覆盖：

- 创建任务未传 `extract_visual` 时默认 `False`
- 前端提交勾选项后，API、`TaskRecord`、job JSON、worker、`ExtractConfig` 全链路保持为 `True`
- `extract_visual=False` 时不调用 `download_video` 和视觉提取器
- 命中下载字幕且 `extract_visual=True` 时仍会下载视频并生成 `pages.json`
- 无字幕且 `extract_visual=True` 时复用视频抽音频，不重复下载 audio
- `meta.json.files` 只在开启视觉提取时登记 `pages_json` 和 `visual_frames`
- `keep_media=False` 时清理 audio/video，但保留 `pages.json` 和代表帧
- OCR 无文字结果时任务成功，`pages` 为空；视频下载或 OCR 运行异常时任务失败

## 落地顺序

1. 先打通任务开关链路：前端 checkbox、API request、任务持久化、worker job、`ExtractConfig`。默认关闭，确保现有测试不受影响。
2. 增加视频复用逻辑：`extract_visual=True` 时确保下载 video；需要 Whisper 时从 video 抽音频。
3. 实现 `FrameExtractor` 和 `pages.json` 空壳输出，先验证文件结构、进度和 meta 登记。
4. 接入 `has_text` 和 OCR，只对文字帧做识别，先允许每个文字段输出一页。
5. 增加 SSIM/pHash 页面变化、稳定检测、去重合并，用实际样本调参。

## 外部依据

- RapidOCR 安装文档：<https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/install/>
- RapidOCR 使用文档：<https://rapidai.github.io/RapidOCRDocs/main/install_usage/rapidocr/usage/>

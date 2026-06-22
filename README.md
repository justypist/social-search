# Social Search

`social-extract` 是搜索前置工具：给定一个视频 URL，提取字幕、转写文本和元信息，供后续搜索索引使用。

## 安装依赖

```bash
uv sync
```

本地转写使用 `faster-whisper`。项目依赖包含 NVIDIA CUDA 12/cuDNN 运行库 wheel；默认会优先尝试 CUDA GPU，如果不可用，会回退到 CPU。

如果需要走“下载视频后提取音频”的兜底路径，系统里需要有 `ffmpeg`：

```bash
ffmpeg -version
```

## 使用

```bash
uv run social-extract "https://example.com/video" --lang auto --model medium --output ./out
```

常用参数：

```bash
uv run social-extract "https://example.com/video" \
  --lang zh \
  --model medium \
  --device auto \
  --output ./out
```

参数说明：

- `--lang auto|zh|en`：默认自动识别，也可指定中文或英文。
- `--model medium`：默认使用 `medium`，也可以传 `small`、`large-v3` 或本地/Hugging Face 模型名。
- `--device auto|cuda|cpu`：默认自动选择。
- `--compute-type auto|float16|int8_float16|int8`：默认自动选择。
- `--vad-filter/--no-vad-filter`：默认关闭 VAD 预过滤；长音频连续讲话时关闭通常更快。
- `--keep-media/--no-keep-media`：默认保留实际下载的音频或视频。
- `--overwrite`：覆盖已存在的输出目录。
- `--add-header "Name:Value"`：额外传给 `yt-dlp` 的 HTTP 请求头，可重复使用。

## Web 前端

首次启动前会读取 `.env`；如果 `.env` 不存在，后端会从 `.env.example` 复制一份默认配置。

安装前端依赖：

```bash
pnpm install
```

启动 Web 和 API：

```bash
pnpm dev
```

默认地址：

- Web/API: `http://127.0.0.1:8000`

常用环境变量在 `.env.example` 中，包含并发数、输出目录、Whisper 模型、设备、语言、是否保留媒体文件等配置。

后端 API 位于 `src/api/`：

- `src/api/main.py`：FastAPI 应用入口、生命周期和路由挂载。
- `src/api/routers/`：HTTP 路由，后续新增 API 优先放这里。
- `src/api/task_manager.py`：提取任务队列、子进程 worker 和文件下载管理。
- `src/api/worker.py`：单个提取任务的子进程执行入口。

## 提取流程

1. 用 `yt-dlp` 探测视频信息，优先下载已有字幕。
2. 如果没有可用字幕，下载最佳音频并用本地 Whisper 转写。
3. 如果音频下载失败，下载视频，用 `ffmpeg` 提取音频，再走 Whisper 转写。

## 输出

默认输出到：

```text
out/
  <video-id-or-safe-title>/
    meta.json
    subtitle.srt
    transcript.txt
    transcript.json
    audio.<fmt>     # 只有实际下载音频时保存
    video.<fmt>     # 只有实际下载视频时保存
```

文件说明：

- `meta.json`：保存原始 URL、视频元信息、提取时间、使用路径、模型、产物路径等。
- `subtitle.srt`：带时间轴的字幕主产物。
- `transcript.txt`：纯文本，适合后续搜索索引。
- `transcript.json`：结构化片段，包含每段的 `start`、`end` 和 `text`。

## 测试链接

https://www.bilibili.com/video/BV1ueLJ61EkJ/?spm_id_from=333.337.search-card.all.click&vd_source=9ef09aa05456714cca397f9a8c5ffc62

https://www.bilibili.com/video/BV1Ec6SBTE9S/?spm_id_from=333.337.search-card.all.click&vd_source=9ef09aa05456714cca397f9a8c5ffc62
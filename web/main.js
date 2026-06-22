const STATUS_LABELS = {
  queued: "排队中",
  running: "运行中",
  stopping: "停止中",
  succeeded: "已完成",
  failed: "已失败",
  cancelled: "已取消",
  stopped: "已停止",
};

const STATUS_TONES = {
  queued: "neutral",
  running: "active",
  stopping: "warning",
  succeeded: "success",
  failed: "danger",
  cancelled: "warning",
  stopped: "warning",
};

const STAGE_LABELS = {
  queued: "等待执行",
  starting: "启动任务",
  probe: "读取视频信息",
  prepare: "准备输出目录",
  subtitle: "处理字幕",
  download_audio: "下载音频",
  download_video: "下载视频",
  extract_audio: "提取音频",
  transcribe: "本地转写",
  write: "写入文件",
  cleanup: "清理媒体",
  done: "完成",
  failed: "失败",
  cancelled: "已取消",
  stopped: "已停止",
  stopping: "停止中",
};

const state = {
  config: null,
  tasks: [],
  selectedId: null,
  isSubmitting: false,
  isRefreshing: false,
  actionTaskId: null,
  error: "",
  confirmDeleteId: null,
  confirmTimer: null,
  themeMode: localStorage.getItem("social-extract-theme") || "system",
};

const elements = {
  configStrip: document.querySelector("#configStrip"),
  queueSummary: document.querySelector("#queueSummary"),
  taskForm: document.querySelector("#taskForm"),
  urlInput: document.querySelector("#urlInput"),
  submitButton: document.querySelector("#submitButton"),
  formError: document.querySelector("#formError"),
  queueMeta: document.querySelector("#queueMeta"),
  taskList: document.querySelector("#taskList"),
  detailMeta: document.querySelector("#detailMeta"),
  detailActions: document.querySelector("#detailActions"),
  taskDetail: document.querySelector("#taskDetail"),
  refreshButton: document.querySelector("#refreshButton"),
  themeSelect: document.querySelector("#themeSelect"),
};

const systemTheme = window.matchMedia("(prefers-color-scheme: dark)");

init();

function init() {
  elements.themeSelect.value = state.themeMode;
  applyTheme();
  systemTheme.addEventListener("change", () => {
    if (state.themeMode === "system") {
      applyTheme();
    }
  });

  elements.themeSelect.addEventListener("change", () => {
    state.themeMode = elements.themeSelect.value;
    localStorage.setItem("social-extract-theme", state.themeMode);
    applyTheme();
  });

  elements.taskForm.addEventListener("submit", handleSubmit);
  elements.refreshButton.addEventListener("click", () => refreshTasks({ quiet: false }));
  elements.taskList.addEventListener("click", handleTaskListClick);
  elements.detailActions.addEventListener("click", handleActionClick);

  loadConfig();
  refreshTasks({ quiet: false });
  window.setInterval(() => refreshTasks({ quiet: true }), 1600);
}

function applyTheme() {
  const resolved = state.themeMode === "system" ? (systemTheme.matches ? "dark" : "light") : state.themeMode;
  document.documentElement.dataset.theme = resolved;
}

async function loadConfig() {
  try {
    const payload = await api("/api/config");
    state.config = payload;
    render();
  } catch (error) {
    state.error = error.message;
    render();
  }
}

async function refreshTasks({ quiet }) {
  if (state.isRefreshing) {
    return;
  }
  state.isRefreshing = true;
  if (!quiet) {
    elements.refreshButton.classList.add("is-loading");
  }

  try {
    const payload = await api("/api/tasks");
    state.tasks = payload.tasks;
    reconcileSelection();
    state.error = "";
  } catch (error) {
    state.error = error.message;
  } finally {
    state.isRefreshing = false;
    elements.refreshButton.classList.remove("is-loading");
    render();
  }
}

async function handleSubmit(event) {
  event.preventDefault();
  const url = elements.urlInput.value.trim();
  elements.formError.textContent = "";
  if (!url) {
    elements.formError.textContent = "请输入视频链接";
    return;
  }

  state.isSubmitting = true;
  renderSubmitState();
  try {
    const payload = await api("/api/tasks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
    elements.urlInput.value = "";
    state.selectedId = payload.task.id;
    await refreshTasks({ quiet: true });
  } catch (error) {
    elements.formError.textContent = error.message;
  } finally {
    state.isSubmitting = false;
    renderSubmitState();
  }
}

function handleTaskListClick(event) {
  const item = event.target.closest("[data-task-id]");
  if (!item) {
    return;
  }
  state.selectedId = item.dataset.taskId;
  resetDeleteConfirm();
  render();
}

async function handleActionClick(event) {
  const button = event.target.closest("[data-action]");
  if (!button || button.disabled) {
    return;
  }
  const taskId = button.dataset.taskId;
  const action = button.dataset.action;

  if (action === "delete" && state.confirmDeleteId !== taskId) {
    state.confirmDeleteId = taskId;
    window.clearTimeout(state.confirmTimer);
    state.confirmTimer = window.setTimeout(resetDeleteConfirm, 3000);
    renderActions(selectedTask());
    return;
  }

  state.actionTaskId = taskId;
  renderActions(selectedTask());
  try {
    if (action === "cancel") {
      await api(`/api/tasks/${taskId}/cancel`, { method: "POST" });
    } else if (action === "stop") {
      await api(`/api/tasks/${taskId}/stop`, { method: "POST" });
    } else if (action === "delete") {
      await api(`/api/tasks/${taskId}`, { method: "DELETE" });
      if (state.selectedId === taskId) {
        state.selectedId = null;
      }
      resetDeleteConfirm();
    }
    await refreshTasks({ quiet: true });
  } catch (error) {
    state.error = error.message;
    render();
  } finally {
    state.actionTaskId = null;
    render();
  }
}

function resetDeleteConfirm() {
  state.confirmDeleteId = null;
  window.clearTimeout(state.confirmTimer);
  state.confirmTimer = null;
  renderActions(selectedTask());
}

function reconcileSelection() {
  if (state.selectedId && state.tasks.some((task) => task.id === state.selectedId)) {
    return;
  }
  state.selectedId = state.tasks[0]?.id || null;
}

function selectedTask() {
  return state.tasks.find((task) => task.id === state.selectedId) || null;
}

function render() {
  renderConfig();
  renderSummary();
  renderSubmitState();
  renderTaskList();
  renderDetail();
}

function renderConfig() {
  if (!state.config) {
    elements.configStrip.textContent = state.error || "配置加载中";
    return;
  }
  elements.configStrip.textContent = `并发 ${state.config.concurrency} | ${state.config.model} | ${state.config.device}`;
}

function renderSummary() {
  const counts = state.tasks.reduce(
    (acc, task) => {
      acc.total += 1;
      acc[task.status] = (acc[task.status] || 0) + 1;
      return acc;
    },
    { total: 0 },
  );
  const running = (counts.running || 0) + (counts.stopping || 0);
  elements.queueSummary.textContent = state.config
    ? `每次执行 ${state.config.concurrency} 个，当前运行 ${running} 个`
    : "队列状态加载中";
  elements.queueMeta.textContent = `${counts.total} 个任务`;
}

function renderSubmitState() {
  elements.submitButton.disabled = state.isSubmitting;
  elements.urlInput.disabled = state.isSubmitting;
  elements.submitButton.classList.toggle("is-loading", state.isSubmitting);
  elements.submitButton.textContent = state.isSubmitting ? "提交中" : "加入队列";
}

function renderTaskList() {
  if (state.error && !state.tasks.length) {
    elements.taskList.innerHTML = `<div class="empty-state error-state">${escapeHtml(state.error)}</div>`;
    return;
  }
  if (!state.tasks.length) {
    elements.taskList.innerHTML = '<div class="empty-state">暂无任务，提交链接后会显示在这里。</div>';
    return;
  }

  elements.taskList.innerHTML = state.tasks.map(renderTaskItem).join("");
}

function renderTaskItem(task) {
  const isSelected = task.id === state.selectedId;
  const tone = STATUS_TONES[task.status] || "neutral";
  return `
    <button class="task-item ${isSelected ? "is-selected" : ""}" type="button" data-task-id="${task.id}">
      <span class="task-row">
        <span class="task-url" title="${escapeHtml(task.url)}">${escapeHtml(task.url)}</span>
        <span class="status-pill tone-${tone}">${STATUS_LABELS[task.status] || task.status}</span>
      </span>
    </button>
  `;
}

function renderDetail() {
  const task = selectedTask();
  renderActions(task);

  if (!task) {
    elements.detailMeta.textContent = "未选择任务";
    elements.taskDetail.innerHTML = '<div class="empty-state">暂无任务详情。</div>';
    return;
  }

  const logScroll = getLogScrollState();
  elements.detailMeta.textContent = task.message || `${STATUS_LABELS[task.status] || task.status} | ${task.progress}%`;
  const errorBlock = task.error ? `<div class="notice danger">${escapeHtml(task.error)}</div>` : "";

  elements.taskDetail.innerHTML = `
    ${errorBlock}
    <div class="detail-stack">
      <section class="detail-section">
        <div class="section-heading">
          <h3>进度</h3>
          <span class="status-pill tone-${STATUS_TONES[task.status] || "neutral"}">${STATUS_LABELS[task.status] || task.status}</span>
        </div>
        <div class="large-progress">
          <span style="width: ${task.progress}%"></span>
        </div>
        <p class="progress-message">${escapeHtml(task.message || stageLabel(task.stage))}</p>
        <dl class="meta-grid">
          <div><dt>阶段</dt><dd>${escapeHtml(stageLabel(task.stage))}</dd></div>
          <div><dt>来源</dt><dd>${escapeHtml(task.source || "-")}</dd></div>
          <div><dt>开始</dt><dd>${formatDate(task.started_at)}</dd></div>
          <div><dt>结束</dt><dd>${formatDate(task.finished_at)}</dd></div>
          <div class="wide"><dt>输出目录</dt><dd>${escapeHtml(task.output_dir || "-")}</dd></div>
          <div class="wide"><dt>链接</dt><dd class="breakable">${escapeHtml(task.url)}</dd></div>
        </dl>
      </section>
      <section class="detail-section">
        <div class="section-heading">
          <h3>日志</h3>
          <span class="muted">${task.logs.length} 条</span>
        </div>
        ${renderLogs(task.logs)}
      </section>
      <section class="detail-section">
        <div class="section-heading">
          <h3>文件</h3>
          <span class="muted">${task.files.length} 个</span>
        </div>
        ${renderFiles(task.files)}
      </section>
    </div>
  `;
  restoreLogScroll(logScroll);
}

function renderActions(task) {
  if (!task) {
    elements.detailActions.innerHTML = "";
    return;
  }

  const busy = state.actionTaskId === task.id;
  const deleteLabel = state.confirmDeleteId === task.id ? "确认删除？" : "删除";
  elements.detailActions.innerHTML = `
    <button class="secondary small" type="button" data-action="cancel" data-task-id="${task.id}" ${!task.can_cancel || busy ? "disabled" : ""}>取消</button>
    <button class="secondary small" type="button" data-action="stop" data-task-id="${task.id}" ${!task.can_stop || busy ? "disabled" : ""}>停止</button>
    <button class="danger small ${state.confirmDeleteId === task.id ? "is-confirming" : ""} ${busy ? "is-loading" : ""}" type="button" data-action="delete" data-task-id="${task.id}" ${!task.can_delete || busy ? "disabled" : ""}>${deleteLabel}</button>
  `;
}

function renderLogs(logs) {
  if (!logs.length) {
    return '<div class="empty-state compact">暂无日志。</div>';
  }
  return `
    <ol class="log-list">
      ${logs
        .map(
          (entry) => `
            <li class="log-entry level-${entry.level}">
              <time>${formatTime(entry.at)}</time>
              <span>${escapeHtml(entry.message)}</span>
            </li>
          `,
        )
        .join("")}
    </ol>
  `;
}

function getLogScrollState() {
  const logList = elements.taskDetail.querySelector(".log-list");
  if (!logList) {
    return { followBottom: true, scrollTop: null };
  }
  const distanceToBottom = logList.scrollHeight - logList.scrollTop - logList.clientHeight;
  return {
    followBottom: distanceToBottom < 32,
    scrollTop: logList.scrollTop,
  };
}

function restoreLogScroll(scrollState) {
  window.requestAnimationFrame(() => {
    const logList = elements.taskDetail.querySelector(".log-list");
    if (!logList) {
      return;
    }
    if (scrollState.followBottom || scrollState.scrollTop === null) {
      logList.scrollTop = logList.scrollHeight;
      return;
    }
    logList.scrollTop = scrollState.scrollTop;
  });
}

function renderFiles(files) {
  if (!files.length) {
    return '<div class="empty-state compact">完成后会显示生成文件。</div>';
  }
  return `
    <div class="file-table-wrap">
      <table class="file-table">
        <thead>
          <tr>
            <th>文件</th>
            <th>大小</th>
            <th>时间</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          ${files
            .map(
              (file) => `
                <tr>
                  <td class="file-name">${escapeHtml(file.name)}</td>
                  <td>${formatSize(file.size)}</td>
                  <td>${formatDate(file.modified_at)}</td>
                  <td><a class="download-link" href="${file.download_url}">下载</a></td>
                </tr>
              `,
            )
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (response.status === 204) {
    return null;
  }
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "object" && payload !== null ? payload.detail : payload;
    throw new Error(Array.isArray(detail) ? "请求参数无效" : detail || "请求失败");
  }
  return payload;
}

function formatDate(value) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "-";
  }
  return new Intl.DateTimeFormat(document.documentElement.lang || "zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

function formatTime(value) {
  if (!value) {
    return "-";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "-";
  }
  return new Intl.DateTimeFormat(document.documentElement.lang || "zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function formatSize(bytes) {
  if (!Number.isFinite(bytes)) {
    return "-";
  }
  return new Intl.NumberFormat(document.documentElement.lang || "zh-CN", {
    maximumFractionDigits: 1,
    style: "unit",
    unit: bytes > 1024 * 1024 ? "megabyte" : "kilobyte",
  }).format(bytes > 1024 * 1024 ? bytes / 1024 / 1024 : Math.max(bytes / 1024, 0.1));
}

function stageLabel(stage) {
  return STAGE_LABELS[stage] || stage || "-";
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

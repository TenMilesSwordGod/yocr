<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { analyze, detect, listModels, ocr } from '../api.js'
import { drawScene, strokeBox, truncate } from '../draw.js'

const MODES = [
  { id: 'detect', label: '元素检测', hint: 'YOLO 检测 UI 元素，返回类别与坐标' },
  { id: 'ocr', label: '文字 OCR', hint: '识别图中全部文字及位置' },
  { id: 'analyze', label: '综合分析', hint: '检测 + OCR，文字归到所属控件' },
]

const mode = ref('detect')
const sceneUrl = ref('')
const sceneFile = ref(null)
const running = ref(false)
const dragging = ref(false)
const error = ref('')
const result = ref(null)
const canvasEl = ref(null)

// params
const models = ref([])
const defaultModel = ref('')
const model = ref('')
const conf = ref(0.25)
const iou = ref(0.45)
const imgsz = ref(1280)
const withOcr = ref(true)
const text = ref('')
const label = ref('')
const q = ref('')
const matchMode = ref('contains')

const hasTarget = computed(() => !!(text.value.trim() || label.value.trim() || q.value.trim()))
const rows = computed(() => {
  if (!result.value) return []
  if (mode.value === 'ocr') return result.value.lines || []
  return result.value.elements || []
})
const rowKey = computed(() => (mode.value === 'ocr' ? 'text' : 'label'))
const modeHint = computed(() => MODES.find(m => m.id === mode.value)?.hint ?? '')

onMounted(async () => {
  try {
    const body = await listModels()
    models.value = body.models
    defaultModel.value = body.default_model
  } catch { /* select stays on server default */ }
})

function setMode(id) {
  mode.value = id
  result.value = null
  error.value = ''
  if (sceneUrl.value) drawPreview()
}

function setSceneFile(file) {
  if (!file || !file.type.startsWith('image/')) return
  sceneFile.value = file
  if (sceneUrl.value) URL.revokeObjectURL(sceneUrl.value)
  sceneUrl.value = URL.createObjectURL(file)
  result.value = null
  error.value = ''
  drawPreview()
}

function pickFile(event) { setSceneFile(event.target.files?.[0]) }
function onDrop(event) {
  dragging.value = false
  setSceneFile(event.dataTransfer?.files?.[0])
}
function onPaste(event) {
  const item = [...(event.clipboardData?.items || [])].find(i => i.type.startsWith('image/'))
  if (item) setSceneFile(item.getAsFile())
}
onMounted(() => window.addEventListener('paste', onPaste))
onBeforeUnmount(() => window.removeEventListener('paste', onPaste))

async function drawPreview() {
  const canvas = canvasEl.value
  if (!canvas || !sceneUrl.value) return
  await drawScene(canvas, sceneUrl.value)
}

async function run() {
  if (!sceneFile.value) { error.value = '请先上传或粘贴图片（截图后 Ctrl+V 即可）'; return }
  running.value = true
  error.value = ''
  try {
    const payload = {
      file: sceneFile.value,
      model: model.value || undefined,
      conf: conf.value,
      iou: iou.value,
      imgsz: imgsz.value,
      text: text.value.trim() || undefined,
      label: label.value.trim() || undefined,
      q: q.value.trim() || undefined,
      matchMode: matchMode.value,
    }
    result.value = mode.value === 'ocr'
      ? await ocr({ file: sceneFile.value })
      : mode.value === 'analyze'
        ? await analyze({ ...payload, withOcr: withOcr.value })
        : await detect(payload)
    await draw()
  } catch (e) {
    error.value = e.message
  } finally {
    running.value = false
  }
}

async function draw() {
  const canvas = canvasEl.value
  if (!canvas || !sceneUrl.value) return
  const ctx = await drawScene(canvas, sceneUrl.value)
  if (!result.value) return

  if (mode.value === 'ocr') {
    for (const line of result.value.lines || []) {
      strokeBox(ctx, line.box.xyxy, {
        color: '#4f8cff',
        width: 1.6,
        label: `${truncate(line.text, 22)} ${(line.confidence * 100).toFixed(0)}%`,
      })
    }
    return
  }
  for (const el of result.value.elements || []) {
    const matched = result.value.matched?.id === el.id
    const textBit = el.text ? ` · ${truncate(el.text, 14)}` : ''
    strokeBox(ctx, el.box.xyxy, {
      color: matched ? '#3dd68c' : '#ffb224',
      width: matched ? 4 : 2,
      label: `${el.label} ${(el.confidence * 100).toFixed(0)}%${textBit}`,
    })
  }
}

async function copyFullText() {
  try {
    await navigator.clipboard.writeText(result.value?.full_text || '')
  } catch { /* clipboard denied — ignore */ }
}
</script>

<template>
  <div class="lab">
    <aside class="card controls" aria-label="分析设置">
      <h2>视觉分析</h2>

      <div class="mode-switch" role="tablist" aria-label="分析模式">
        <button
          v-for="m in MODES"
          :key="m.id"
          role="tab"
          :aria-selected="mode === m.id"
          class="mode-tab"
          :class="{ active: mode === m.id }"
          @click="setMode(m.id)"
        >{{ m.label }}</button>
      </div>
      <p class="hint">{{ modeHint }}</p>

      <div
        class="dropzone"
        :class="{ filled: sceneUrl, dragging }"
        @dragover.prevent="dragging = true"
        @dragleave="dragging = false"
        @drop="onDrop"
      >
        <div v-if="!sceneUrl" class="hint">
          <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
            <rect x="3" y="3" width="18" height="18" rx="3"/>
            <path d="M12 8v8M8 12h8"/>
          </svg>
          拖拽 / 点击上传<br />或直接 <b>Ctrl+V</b> 粘贴截图
        </div>
        <div v-else class="picked">已选择图片，可重新上传替换</div>
        <input type="file" accept="image/*" aria-label="选择分析图片" @change="pickFile" />
      </div>

      <template v-if="mode !== 'ocr'">
        <label class="field">
          <span>模型</span>
          <select v-model="model">
            <option value="">默认（{{ defaultModel || '服务器默认' }}）</option>
            <option v-for="m in models" :key="m.name" :value="m.name">
              {{ m.name }}{{ m.loaded ? ' ●' : ' ○' }}
            </option>
          </select>
        </label>
        <label class="field">
          <span>置信度 conf：<b>{{ conf.toFixed(2) }}</b></span>
          <input v-model.number="conf" type="range" min="0.05" max="0.95" step="0.05" />
        </label>
        <label v-if="mode === 'detect'" class="field">
          <span>NMS IoU：<b>{{ iou.toFixed(2) }}</b></span>
          <input v-model.number="iou" type="range" min="0.1" max="0.9" step="0.05" />
        </label>
        <label class="field">
          <span>推理分辨率 imgsz</span>
          <select v-model.number="imgsz">
            <option :value="640">640（最快）</option>
            <option :value="960">960</option>
            <option :value="1280">1280（默认）</option>
            <option :value="1600">1600（最准）</option>
          </select>
        </label>
        <label v-if="mode === 'analyze'" class="check">
          <input v-model="withOcr" type="checkbox" />
          <span>同时运行 OCR 并把文字归到控件</span>
        </label>

        <details class="target">
          <summary>查找目标（可选）</summary>
          <label class="field">
            <span>按 OCR 文本找 text</span>
            <input v-model="text" placeholder="如 设置" spellcheck="false" />
          </label>
          <label class="field">
            <span>按类别名找 label</span>
            <input v-model="label" placeholder="如 Button" spellcheck="false" />
          </label>
          <label class="field">
            <span>泛搜索 q（文本或类别任一命中）</span>
            <input v-model="q" placeholder="如 bluetooth" spellcheck="false" />
          </label>
          <label class="field">
            <span>匹配方式</span>
            <select v-model="matchMode">
              <option value="contains">包含 contains</option>
              <option value="exact">精确 exact</option>
              <option value="fuzzy">模糊 fuzzy</option>
            </select>
          </label>
        </details>
      </template>

      <button class="run-btn" :disabled="running" @click="run">
        <span v-if="running" class="spinner" aria-hidden="true" />
        <svg v-else viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
        {{ running ? '分析中…' : '开始分析' }}
      </button>

      <p v-if="error" class="msg error" role="alert">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/></svg>
        {{ error }}
      </p>

      <p v-if="result" class="dim meta-line">
        {{ result.model ? `${result.model} · ` : '' }}{{ result.timing.total_ms }} ms · 图 {{ result.image.width }}×{{ result.image.height }}
        <template v-if="mode !== 'ocr' && hasTarget">
          · <b :class="result.found ? 'found' : 'miss'">{{ result.found ? `命中：${result.matched?.label}` : '未命中目标' }}</b>
        </template>
      </p>
    </aside>

    <section class="stage-area">
      <div class="stage card" :class="{ scanning: running }">
        <canvas v-show="sceneUrl" ref="canvasEl" class="canvas" />
        <div v-if="!sceneUrl" class="placeholder">
          <svg viewBox="0 0 96 64" width="130" aria-hidden="true">
            <rect x="10" y="8" width="76" height="48" rx="6" fill="none" stroke="var(--border-strong)" stroke-width="2"/>
            <rect x="26" y="20" width="20" height="14" rx="2" fill="none" stroke="var(--warn)" stroke-width="2"/>
            <rect x="54" y="30" width="22" height="12" rx="2" fill="none" stroke="var(--accent)" stroke-width="2"/>
            <path d="M30 44h36" stroke="var(--border-strong)" stroke-width="2" stroke-linecap="round"/>
          </svg>
          <p>上传截图后选择分析模式，元素和文字会框在这里</p>
        </div>
        <div v-if="running" class="scanline" aria-hidden="true" />
      </div>

      <div v-if="result" class="results">
        <div class="results-head">
          <h3>{{ mode === 'ocr' ? `文字行（${rows.length}）` : `元素（${rows.length}）` }}</h3>
          <button v-if="mode === 'ocr' && result.full_text" class="ghost copy" @click="copyFullText">
            <svg viewBox="0 0 24 24" width="13" height="13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>
            复制全文
          </button>
        </div>

        <pre v-if="mode === 'ocr' && result.full_text" class="fulltext">{{ result.full_text }}</pre>

        <div class="rows">
          <div v-for="(r, i) in rows" :key="i" class="row-item">
            <template v-if="mode === 'ocr'">
              <span class="row-main">{{ r.text }}</span>
              <span class="row-conf">{{ (r.confidence * 100).toFixed(0) }}%</span>
            </template>
            <template v-else>
              <span class="row-label" :class="{ matched: result.matched?.id === r.id }">{{ r.label }}</span>
              <span class="row-conf">{{ (r.confidence * 100).toFixed(0) }}%</span>
              <span class="row-box">({{ r.box.xyxy.join(', ') }})</span>
              <span v-if="r.text" class="row-text" :title="r.text">{{ r.text }}</span>
            </template>
          </div>
          <p v-if="!rows.length" class="dim empty-rows">没有{{ mode === 'ocr' ? '识别到文字' : '检测到元素' }}，试试调低置信度。</p>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.lab {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 22px;
  align-items: start;
}
@media (max-width: 960px) {
  .lab { grid-template-columns: 1fr; }
}
.controls h2 { margin: 0 0 14px; font-size: 16px; font-weight: 600; }

.mode-switch {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 4px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 3px;
  margin-bottom: 8px;
}
.mode-tab {
  min-height: 30px;
  padding: 0 6px;
  border-radius: 8px;
  border-color: transparent;
  background: transparent;
  color: var(--text-dim);
  font-size: 13px;
}
.mode-tab:hover:not(:disabled) { background: transparent; color: var(--text); }
.mode-tab.active { background: var(--accent); color: #fff; }
.hint { margin: 0 0 14px; font-size: 12px; color: var(--text-faint); }

.dropzone {
  position: relative;
  border: 1.5px dashed var(--border-strong);
  border-radius: 10px;
  min-height: 92px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 14px;
  background: var(--panel-2);
  transition: border-color var(--t-fast) var(--ease), background var(--t-fast) var(--ease);
}
.dropzone:hover, .dropzone.dragging { border-color: var(--accent); background: var(--accent-subtle); }
.dropzone .hint {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: var(--text-dim);
  font-size: 13px;
  text-align: center;
  line-height: 1.7;
  pointer-events: none;
}
.dropzone .hint b { color: var(--text); }
.dropzone .picked { color: var(--text-dim); font-size: 13px; padding: 12px; text-align: center; }
.dropzone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }

.check {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-bottom: 14px;
  cursor: pointer;
  font-size: 13px;
  color: var(--text-dim);
}
.check:hover { color: var(--text); }

.target {
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 10px 12px;
  margin-bottom: 14px;
}
.target summary {
  cursor: pointer;
  font-size: 13px;
  color: var(--text-dim);
  user-select: none;
}
.target[open] summary { margin-bottom: 12px; color: var(--text); }
.target .field { margin-bottom: 10px; }
.target .field:last-child { margin-bottom: 0; }

.run-btn { width: 100%; }
.msg { margin-bottom: 12px; }
.dim { color: var(--text-dim); font-size: 12px; }
.meta-line { margin: 12px 0 0; }
.found { color: var(--ok); }
.miss { color: var(--warn); }

.stage {
  position: relative;
  padding: 8px;
  min-height: 260px;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
}
.canvas { width: 100%; height: auto; display: block; border-radius: 6px; }
.placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  color: var(--text-dim);
  font-size: 13px;
  padding: 40px 20px;
  text-align: center;
}
.stage.scanning .canvas { opacity: .8; }
.scanline {
  position: absolute;
  left: 8px;
  right: 8px;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent) 18%, var(--accent) 82%, transparent);
  animation: scan 1.5s var(--ease) infinite alternate;
  pointer-events: none;
}
@keyframes scan { from { top: 6%; } to { top: 92%; } }

.results { margin-top: 22px; }
.results-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px; }
.results-head h3 { font-size: 13px; color: var(--text-dim); font-weight: 500; margin: 0; }
.copy { min-height: 28px; padding: 0 10px; font-size: 12px; }
.fulltext {
  margin: 0 0 12px;
  padding: 12px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
  font-size: 12px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-all;
  max-height: 200px;
  overflow: auto;
  color: var(--text);
}
.rows { display: grid; gap: 8px; }
.row-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 12px;
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  font-size: 13px;
  flex-wrap: wrap;
}
.row-main { flex: 1; min-width: 0; }
.row-label {
  background: var(--warn-subtle);
  color: var(--warn);
  border-radius: 999px;
  padding: 1px 9px;
  font-size: 12px;
  font-weight: 600;
}
.row-label.matched { background: var(--ok-subtle); color: var(--ok); }
.row-conf { color: var(--text-dim); font-size: 12px; }
.row-box { color: var(--text-faint); font-size: 12px; font-family: ui-monospace, Consolas, monospace; }
.row-text {
  color: var(--text-dim);
  font-size: 12px;
  max-width: 240px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.empty-rows { padding: 12px; }

@media (prefers-reduced-motion: reduce) {
  .scanline { display: none; }
}
</style>

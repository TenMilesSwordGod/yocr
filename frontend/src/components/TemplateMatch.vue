<script setup>
import { onBeforeUnmount, onMounted, ref } from 'vue'
import { matchTemplate } from '../api.js'
import { drawScene, strokeBox } from '../draw.js'

const sceneUrl = ref('')
const sceneFile = ref(null)
const templateUrl = ref('')
const templateFile = ref(null)
const threshold = ref(0.8)
const running = ref(false)
const error = ref('')
const result = ref(null)
const canvasEl = ref(null)
const draggingScene = ref(false)
const draggingTpl = ref(false)

function makeSlot(fileRef, urlRef, draggingRef) {
  function set(file) {
    if (!file || !file.type.startsWith('image/')) return
    fileRef.value = file
    if (urlRef.value) URL.revokeObjectURL(urlRef.value)
    urlRef.value = URL.createObjectURL(file)
    result.value = null
    error.value = ''
    drawPreview()
  }
  return {
    pick: (event) => set(event.target.files?.[0]),
    drop: (event) => {
      draggingRef.value = false
      set(event.dataTransfer?.files?.[0])
    },
  }
}

const scene = makeSlot(sceneFile, sceneUrl, draggingScene)
const tpl = makeSlot(templateFile, templateUrl, draggingTpl)

// Ctrl+V fills the scene slot (template is a small crop you pick from disk)
function onPaste(event) {
  const item = [...(event.clipboardData?.items || [])].find(i => i.type.startsWith('image/'))
  if (item) scene.pick({ target: { files: [item.getAsFile()] } })
}
onMounted(() => window.addEventListener('paste', onPaste))
onBeforeUnmount(() => window.removeEventListener('paste', onPaste))

async function drawPreview() {
  const canvas = canvasEl.value
  if (!canvas || !sceneUrl.value) return
  const ctx = await drawScene(canvas, sceneUrl.value)
  if (result.value?.found && result.value.box) {
    strokeBox(ctx, result.value.box.xyxy, {
      color: '#3dd68c',
      width: 4,
      label: `${Number(result.value.score).toFixed(2)} @${result.value.scale}x`,
    })
  }
}

async function run() {
  if (!sceneFile.value || !templateFile.value) {
    error.value = '请先提供场景大图和模板小图两张图片'
    return
  }
  running.value = true
  error.value = ''
  try {
    result.value = await matchTemplate({
      sceneFile: sceneFile.value,
      templateFile: templateFile.value,
      threshold: threshold.value,
    })
    await drawPreview()
  } catch (e) {
    error.value = e.message
  } finally {
    running.value = false
  }
}
</script>

<template>
  <div class="match">
    <aside class="card controls" aria-label="模板匹配设置">
      <h2>模板匹配</h2>
      <p class="hint">上传一张大图（场景）和一张小图（模板），毫秒级定位小图在大图中的位置。与「查找定位」的区别：这里不入库，即传即比。</p>

      <label class="field">
        <span>场景大图（在此图中查找）</span>
        <div
          class="dropzone"
          :class="{ filled: sceneUrl, dragging: draggingScene }"
          @dragover.prevent="draggingScene = true"
          @dragleave="draggingScene = false"
          @drop="scene.drop"
        >
          <div v-if="!sceneUrl" class="hint">场景截图<br />可 Ctrl+V 粘贴</div>
          <div v-else class="picked">已选择</div>
          <input type="file" accept="image/*" aria-label="选择场景大图" @change="scene.pick" />
        </div>
      </label>

      <label class="field">
        <span>模板小图（要找的目标）</span>
        <div
          class="dropzone small"
          :class="{ filled: templateUrl, dragging: draggingTpl }"
          @dragover.prevent="draggingTpl = true"
          @dragleave="draggingTpl = false"
          @drop="tpl.drop"
        >
          <div v-if="!templateUrl" class="hint">按钮 / 图标小图</div>
          <div v-else class="picked">已选择</div>
          <input type="file" accept="image/*" aria-label="选择模板小图" @change="tpl.pick" />
        </div>
      </label>

      <label class="field">
        <span>判定阈值：<b>{{ threshold.toFixed(2) }}</b></span>
        <input v-model.number="threshold" type="range" min="0.5" max="0.98" step="0.01" />
      </label>

      <button class="run-btn" :disabled="running || !sceneFile || !templateFile" @click="run">
        <span v-if="running" class="spinner" aria-hidden="true" />
        <svg v-else viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
        {{ running ? '匹配中…' : '开始匹配' }}
      </button>

      <p v-if="error" class="msg error" role="alert">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/></svg>
        {{ error }}
      </p>

      <div v-if="result" class="summary">
        <p class="msg" :class="result.found ? 'ok' : 'warn'" role="status">
          <svg v-if="result.found" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
          <svg v-else viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/></svg>
          {{ result.found ? `找到模板，得分 ${result.score}` : `未找到（最高分 ${result.score}）` }}
        </p>
        <p v-if="result.found" class="dim">
          中心 ({{ result.box.center[0] }}, {{ result.box.center[1] }}) · 缩放 {{ result.scale }}x · {{ result.timing.total_ms }} ms
        </p>
      </div>
    </aside>

    <section class="stage-area">
      <div class="stage card" :class="{ scanning: running }">
        <canvas v-show="sceneUrl" ref="canvasEl" class="canvas" />
        <div v-if="!sceneUrl" class="placeholder">
          <svg viewBox="0 0 96 64" width="130" aria-hidden="true">
            <rect x="8" y="10" width="56" height="44" rx="5" fill="none" stroke="var(--border-strong)" stroke-width="2"/>
            <rect x="66" y="24" width="22" height="16" rx="3" fill="none" stroke="var(--accent)" stroke-width="2"/>
            <path d="M64 32h-6" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-dasharray="3 3"/>
          </svg>
          <p>场景图会显示在这里，命中的模板位置用绿框标出</p>
        </div>
        <div v-if="running" class="scanline" aria-hidden="true" />
      </div>
    </section>
  </div>
</template>

<style scoped>
.match {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 22px;
  align-items: start;
}
@media (max-width: 960px) {
  .match { grid-template-columns: 1fr; }
}
.controls h2 { margin: 0 0 10px; font-size: 16px; font-weight: 600; }
.hint { margin: 0 0 16px; font-size: 12px; color: var(--text-faint); line-height: 1.7; }

.dropzone {
  position: relative;
  border: 1.5px dashed var(--border-strong);
  border-radius: 10px;
  min-height: 88px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--panel-2);
  transition: border-color var(--t-fast) var(--ease), background var(--t-fast) var(--ease);
}
.dropzone:hover, .dropzone.dragging { border-color: var(--accent); background: var(--accent-subtle); }
.dropzone .hint { color: var(--text-dim); font-size: 13px; text-align: center; line-height: 1.7; pointer-events: none; }
.dropzone .picked { color: var(--text-dim); font-size: 13px; }
.dropzone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }

.run-btn { width: 100%; margin-top: 4px; }
.msg { margin-bottom: 12px; }
.dim { color: var(--text-dim); font-size: 12px; }
.summary { margin-top: 14px; }

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

@media (prefers-reduced-motion: reduce) {
  .scanline { display: none; }
}
</style>

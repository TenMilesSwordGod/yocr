<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { findSucai, sucaiImageUrl } from '../api.js'

const sceneUrl = ref('')
const sceneFile = ref(null)
const threshold = ref(0.8)
const allInstances = ref(true)
const searching = ref(false)
const error = ref('')
const response = ref(null)
const selected = ref(null)
const canvasEl = ref(null)
const dragging = ref(false)

const foundResults = computed(() => (response.value?.results || []).filter(r => r.found))
const bestScore = computed(() => {
  const scores = response.value?.results?.map(r => r.score) ?? []
  return scores.length ? Math.max(...scores) : null
})
const totalHits = computed(() =>
  foundResults.value.reduce((n, r) => n + (r.hits?.length || 1), 0)
)

function setSceneFile(file) {
  if (!file || !file.type.startsWith('image/')) return
  sceneFile.value = file
  if (sceneUrl.value) URL.revokeObjectURL(sceneUrl.value)
  sceneUrl.value = URL.createObjectURL(file)
  response.value = null
  selected.value = null
  // preview the scene on the stage right away, boxes come after compare
  draw()
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

async function search() {
  if (!sceneFile.value) { error.value = '请先上传或粘贴期望图片（截图后 Ctrl+V 即可）'; return }
  searching.value = true
  error.value = ''
  try {
    response.value = await findSucai({
      sceneFile: sceneFile.value,
      threshold: threshold.value,
      allInstances: allInstances.value,
    })
    selected.value = response.value.results.find(r => r.found) || null
    await draw()
  } catch (e) {
    error.value = e.message
  } finally {
    searching.value = false
  }
}

function select(result) {
  selected.value = result
  draw()
}

async function draw() {
  const canvas = canvasEl.value
  if (!canvas || !sceneUrl.value) return
  const img = new Image()
  await new Promise((resolve, reject) => { img.onload = resolve; img.onerror = reject; img.src = sceneUrl.value })
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  const ctx = canvas.getContext('2d')
  ctx.drawImage(img, 0, 0)

  if (!response.value) return
  for (const r of foundResults.value) {
    // all_instances mode reports every occurrence in r.hits; otherwise fall
    // back to the single best box.
    const instances = r.hits?.length ? r.hits : (r.box ? [{ box: r.box, score: r.score }] : [])
    instances.forEach((inst, idx) => {
      if (!inst.box) return
      const isSel = selected.value?.id === r.id
      const active = !selected.value || isSel
      ctx.save()
      ctx.lineWidth = isSel ? 4 : 2.5
      ctx.strokeStyle = active ? (isSel ? '#3dd68c' : '#ffb224') : 'rgba(255,178,36,.35)'
      ctx.setLineDash(active ? [] : [6, 4])
      ctx.strokeRect(...inst.box.xyxy)
      // label chip
      const label = `${r.describe || r.id}${instances.length > 1 ? ` #${idx + 1}` : ''} ${inst.score}`
      ctx.font = '14px sans-serif'
      const w = ctx.measureText(label).width + 10
      const ly = Math.max(inst.box.xyxy[1] - 22, 0)
      ctx.fillStyle = ctx.strokeStyle
      ctx.fillRect(inst.box.xyxy[0], ly, w, 20)
      ctx.fillStyle = '#10131a'
      ctx.fillText(label, inst.box.xyxy[0] + 5, ly + 15)
      ctx.restore()
    })
  }
}
</script>

<template>
  <div class="finder">
    <aside class="card controls" aria-label="比对设置">
      <h2>期望图片</h2>
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
        <input type="file" accept="image/*" aria-label="选择期望图片" @change="pickFile" />
      </div>

      <label class="field">
        <span>判定阈值：<b>{{ threshold.toFixed(2) }}</b></span>
        <input v-model.number="threshold" type="range" min="0.5" max="0.98" step="0.01" />
      </label>

      <label class="check">
        <input v-model="allInstances" type="checkbox" />
        <span>标记所有出现位置（同一素材出现多次时全部框出）</span>
      </label>

      <button class="search-btn" :disabled="searching" @click="search">
        <span v-if="searching" class="spinner" aria-hidden="true" />
        <svg v-else viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg>
        {{ searching ? '比对中…' : response ? '重新比对' : '开始比对' }}
      </button>

      <p v-if="error" class="msg error" role="alert">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/></svg>
        {{ error }}
      </p>

      <div v-if="response" class="summary">
        <p class="msg" :class="response.found_any ? 'ok' : 'warn'" role="status">
          <svg v-if="response.found_any" viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
          <svg v-else viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/></svg>
          <span>
            {{ response.found_any
              ? `命中 ${foundResults.length} 个素材${totalHits > foundResults.length ? `（共 ${totalHits} 处）` : ''}`
              : `未命中任何素材${bestScore !== null ? `，最高分 ${bestScore}` : ''}` }}
          </span>
        </p>
        <p class="dim">{{ response.sucai_count }} 个素材 · {{ response.timing.total_ms }} ms · 图 {{ response.image.width }}×{{ response.image.height }}</p>
      </div>
    </aside>

    <section class="stage-area">
      <div class="stage card" :class="{ scanning: searching }">
        <canvas v-show="sceneUrl" ref="canvasEl" class="canvas" />
        <div v-if="!sceneUrl" class="placeholder">
          <svg viewBox="0 0 96 64" width="130" aria-hidden="true">
            <rect x="10" y="8" width="76" height="48" rx="6" fill="none" stroke="var(--border-strong)" stroke-width="2"/>
            <rect x="30" y="22" width="36" height="20" rx="3" fill="none" stroke="var(--accent)" stroke-width="2" stroke-dasharray="4 4"/>
            <circle cx="48" cy="32" r="2.5" fill="var(--accent)"/>
          </svg>
          <p>上传期望图片后，这里会框出每个素材的位置</p>
        </div>
        <div v-if="searching" class="scanline" aria-hidden="true" />
      </div>

      <div v-if="response" class="results">
        <h3>比对结果（按得分排序）</h3>
        <div class="result-grid">
          <article
            v-for="r in response.results"
            :key="r.id"
            class="card result"
            :class="{ sel: selected?.id === r.id }"
            role="button"
            tabindex="0"
            :aria-pressed="selected?.id === r.id"
            :aria-label="`素材 ${r.id}，得分 ${r.score}，${r.found ? '命中' : '未中'}`"
            @click="select(r)"
            @keydown.enter.prevent="select(r)"
            @keydown.space.prevent="select(r)"
          >
            <img :src="sucaiImageUrl(r.id)" :alt="r.id" />
            <div class="info">
              <div class="tags">
                <code>{{ r.id }}</code>
                <span v-if="r.category" class="cat">{{ r.category }}</span>
              </div>
              <p class="desc">{{ r.describe || '—' }}</p>
              <div class="scorebar" :title="`阈值 ${threshold.toFixed(2)}`">
                <i :style="{ transform: `scaleX(${Math.max(0, Math.min(1, r.score))})` }" />
                <mark class="mark" :style="{ left: (threshold * 100) + '%' }" aria-hidden="true" />
              </div>
              <div class="row">
                <span class="score">{{ r.score.toFixed(4) }}</span>
                <span class="badge" :class="r.found ? 'hit' : 'miss'">{{ r.found ? '命中' : '未中' }}</span>
                <span v-if="r.hits && r.hits.length > 1" class="badge multi">×{{ r.hits.length }} 处</span>
                <span v-if="r.center" class="dim">@ ({{ r.center[0] }}, {{ r.center[1] }})</span>
              </div>
            </div>
          </article>
        </div>
      </div>
    </section>
  </div>
</template>

<style scoped>
.finder {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 22px;
  align-items: start;
}
@media (max-width: 960px) {
  .finder { grid-template-columns: 1fr; }
}
.controls h2 { margin: 0 0 16px; font-size: 16px; font-weight: 600; }

.dropzone {
  position: relative;
  border: 1.5px dashed var(--border-strong);
  border-radius: 10px;
  min-height: 96px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 16px;
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
  margin-bottom: 16px;
  cursor: pointer;
  font-size: 13px;
  color: var(--text-dim);
}
.check span { line-height: 1.5; }
.check:hover { color: var(--text); }

.search-btn { width: 100%; }
.msg { margin-bottom: 12px; }
.dim { color: var(--text-dim); font-size: 12px; }
.summary { margin-top: 14px; }

/* stage: the hero */
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

/* the one authored moment: a scan sweep while comparing */
.scanline {
  position: absolute;
  left: 8px;
  right: 8px;
  top: 10%;
  height: 2px;
  background: linear-gradient(90deg, transparent, var(--accent) 18%, var(--accent) 82%, transparent);
  animation: scan 1.5s var(--ease) infinite alternate;
  pointer-events: none;
}
.scanline::after {
  content: '';
  position: absolute;
  inset: 2px 0 auto;
  height: 34px;
  background: linear-gradient(180deg, var(--accent-subtle), transparent);
}
@keyframes scan { from { top: 6%; } to { top: 92%; } }

.results { margin-top: 22px; }
.results h3 {
  font-size: 13px;
  color: var(--text-dim);
  font-weight: 500;
  margin: 0 0 10px;
}
.result-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(235px, 1fr));
  gap: 12px;
}
.result {
  display: flex;
  gap: 10px;
  cursor: pointer;
  padding: 10px;
  transition: border-color var(--t-fast) var(--ease), box-shadow var(--t-med) var(--ease);
}
.result:hover { border-color: var(--border-strong); }
.result.sel { border-color: var(--ok); box-shadow: 0 0 0 1px var(--ok), var(--shadow-1); }
.result img {
  width: 62px;
  height: 62px;
  object-fit: contain;
  background: #fff;
  border-radius: 6px;
  flex-shrink: 0;
}
.info { min-width: 0; flex: 1; }
.tags { display: flex; align-items: center; gap: 6px; flex-wrap: wrap; }
.info code { font-size: 12px; color: var(--accent); word-break: break-all; }
.cat {
  font-size: 11px;
  color: var(--accent);
  background: var(--accent-subtle);
  border-radius: 999px;
  padding: 0 7px;
  white-space: nowrap;
}
.desc { margin: 4px 0; font-size: 12px; color: var(--text-dim); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.scorebar {
  position: relative;
  height: 4px;
  border-radius: 2px;
  background: var(--panel-2);
  overflow: visible;
  margin: 7px 0;
}
.scorebar i {
  display: block;
  width: 100%;
  height: 100%;
  border-radius: 2px;
  background: linear-gradient(90deg, var(--warn), var(--ok));
  transform-origin: left center;
  transition: transform var(--t-med) var(--ease);
}
.scorebar .mark {
  position: absolute;
  top: -3px;
  width: 2px;
  height: 10px;
  background: var(--text-faint);
  border-radius: 1px;
}
.row { display: flex; align-items: center; gap: 7px; font-size: 11px; flex-wrap: wrap; }
.score { color: var(--text-dim); }
.badge { border-radius: 999px; padding: 1px 8px; font-weight: 600; }
.badge.hit { background: var(--ok-subtle); color: var(--ok); }
.badge.miss { background: rgba(162, 169, 182, .14); color: var(--text-dim); }
.badge.multi { background: var(--accent-subtle); color: var(--accent); }

@media (prefers-reduced-motion: reduce) {
  .scanline { display: none; }
}
</style>

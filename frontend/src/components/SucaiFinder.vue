<script setup>
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { findSucai, sucaiImageUrl } from '../api.js'

const sceneUrl = ref('')
const sceneFile = ref(null)
const sceneInfo = ref(null)
const threshold = ref(0.8)
const allInstances = ref(true)
const searching = ref(false)
const error = ref('')
const response = ref(null)
const selected = ref(null)
const canvasEl = ref(null)

const foundResults = computed(() => (response.value?.results || []).filter(r => r.found))
const bestScore = computed(() => {
  const scores = response.value?.results?.map(r => r.score) ?? []
  return scores.length ? Math.max(...scores) : null
})

function setSceneFile(file) {
  if (!file || !file.type.startsWith('image/')) return
  sceneFile.value = file
  if (sceneUrl.value) URL.revokeObjectURL(sceneUrl.value)
  sceneUrl.value = URL.createObjectURL(file)
  response.value = null
  selected.value = null
}

function pickFile(event) { setSceneFile(event.target.files?.[0]) }
function onDrop(event) {
  event.preventDefault()
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
      ctx.strokeStyle = active ? (isSel ? '#34c77b' : '#ffb020') : 'rgba(255,176,32,.35)'
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
    <aside class="card controls">
      <h2>期望图片</h2>
      <div class="dropzone" :class="{ filled: sceneUrl }" @dragover.prevent @drop="onDrop">
        <div v-if="!sceneUrl" class="hint">
          拖拽 / 点击上传<br />或直接 <b>Ctrl+V</b> 粘贴截图
        </div>
        <input type="file" accept="image/*" @change="pickFile" />
      </div>

      <label class="field">
        <span>判定阈值：<b>{{ threshold.toFixed(2) }}</b></span>
        <input v-model.number="threshold" type="range" min="0.5" max="0.98" step="0.01" />
      </label>

      <label class="check">
        <input v-model="allInstances" type="checkbox" />
        <span>标记所有出现位置（同一素材出现多次时全部框出）</span>
      </label>

      <button :disabled="searching" style="width:100%" @click="search">
        {{ searching ? '比对中…' : `开始比对${response ? '（重新）' : ''}` }}
      </button>

      <p v-if="error" class="msg error">{{ error }}</p>

      <div v-if="response" class="summary">
        <p class="msg" :class="response.found_any ? 'ok' : 'warn'">
          {{ response.found_any
            ? `命中 ${foundResults.length} 个素材`
            : `未命中任何素材${bestScore !== null ? `（最高分 ${bestScore}）` : ''}` }}
        </p>
        <p class="dim">{{ response.sucai_count }} 个素材 · {{ response.timing.total_ms }} ms · 图 {{ response.image.width }}×{{ response.image.height }}</p>
      </div>
    </aside>

    <section class="stage-area">
      <div v-if="sceneUrl" class="stage card">
        <canvas ref="canvasEl" class="canvas" />
      </div>
      <div v-else class="stage card placeholder">上传期望图片后，这里会显示定位结果</div>

      <div v-if="response" class="results">
        <h3>比对结果（按得分排序）</h3>
        <div class="result-grid">
          <article
            v-for="r in response.results"
            :key="r.id"
            class="card result"
            :class="{ sel: selected?.id === r.id }"
            @click="select(r)"
          >
            <img :src="sucaiImageUrl(r.id)" :alt="r.id" />
            <div class="info">
              <code>{{ r.id }}</code>
              <p class="desc">{{ r.describe || '—' }}</p>
              <div class="scorebar"><i :style="{ width: (Math.max(0, r.score) * 100) + '%' }" /></div>
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
  gap: 20px;
  align-items: start;
}
@media (max-width: 960px) {
  .finder { grid-template-columns: 1fr; }
}
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
}
.controls h2 { margin: 0 0 14px; font-size: 16px; }

.dropzone {
  position: relative;
  border: 1px dashed var(--border);
  border-radius: 10px;
  min-height: 90px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 14px;
  background: var(--panel-2);
}
.dropzone:hover { border-color: var(--accent); }
.dropzone .hint { color: var(--text-dim); font-size: 13px; text-align: center; line-height: 1.7; }
.dropzone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }

input[type=range] { accent-color: var(--accent); }

.msg { border-radius: 8px; padding: 8px 12px; font-size: 13px; }
.msg.error { background: rgba(255,93,93,.12); color: var(--bad); }
.msg.ok { background: rgba(52,199,123,.12); color: var(--ok); }
.msg.warn { background: rgba(255,176,32,.12); color: var(--warn); }
.dim { color: var(--text-dim); font-size: 12px; }
.summary { margin-top: 12px; }

.stage { padding: 8px; }
.stage.placeholder {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 260px;
  color: var(--text-dim);
}
.canvas { width: 100%; height: auto; display: block; border-radius: 6px; }

.results { margin-top: 20px; }
.results h3 { font-size: 14px; color: var(--text-dim); font-weight: 500; }
.result-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(230px, 1fr));
  gap: 12px;
}
.result { display: flex; gap: 10px; cursor: pointer; transition: border-color .15s; padding: 10px; }
.result:hover { border-color: var(--text-dim); }
.result.sel { border-color: var(--ok); }
.result img {
  width: 64px;
  height: 64px;
  object-fit: contain;
  background: #fff;
  border-radius: 6px;
  flex-shrink: 0;
}
.info { min-width: 0; flex: 1; }
.info code { font-size: 12px; color: var(--accent); word-break: break-all; }
.desc { margin: 4px 0; font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.scorebar { height: 4px; border-radius: 2px; background: var(--panel-2); overflow: hidden; margin: 6px 0; }
.scorebar i { display: block; height: 100%; background: linear-gradient(90deg, var(--warn), var(--ok)); }
.row { display: flex; align-items: center; gap: 8px; font-size: 11px; }
.score { color: var(--text-dim); }
.badge { border-radius: 999px; padding: 1px 8px; font-weight: 600; }
.badge.hit { background: rgba(52,199,123,.15); color: var(--ok); }
.badge.miss { background: rgba(154,160,171,.15); color: var(--text-dim); }
.badge.multi { background: rgba(79,140,255,.15); color: var(--accent); }
.check {
  display: flex;
  align-items: flex-start;
  gap: 8px;
  margin-bottom: 14px;
  cursor: pointer;
  font-size: 13px;
  color: var(--text-dim);
}
.check input { width: auto; accent-color: var(--accent); margin-top: 2px; }
.check:hover { color: var(--text); }
</style>

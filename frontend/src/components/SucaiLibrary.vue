<script setup>
import { computed, onMounted, ref } from 'vue'
import { createSucai, deleteSucai, listSucai, listSucaiCategories, sucaiImageUrl } from '../api.js'

const items = ref([])
const total = ref(0)
const categories = ref([])
const loading = ref(false)
const error = ref('')
const notice = ref('')
let noticeTimer = null

// list state: category filter + pagination
const categoryFilter = ref('')
const page = ref(1)
const pageSize = ref(24)
const totalPages = computed(() => Math.max(1, Math.ceil(total.value / pageSize.value)))

// register form state
const idInput = ref('')
const describeInput = ref('')
const categoryInput = ref('')
const pickedFile = ref(null)
const previewUrl = ref('')
const dragging = ref(false)
const submitting = ref(false)

function flashNotice(text) {
  notice.value = text
  clearTimeout(noticeTimer)
  noticeTimer = setTimeout(() => { notice.value = '' }, 3200)
}

async function refreshCategories() {
  try {
    categories.value = (await listSucaiCategories()).categories
  } catch { /* non-fatal */ }
}

async function refresh() {
  loading.value = true
  error.value = ''
  try {
    const body = await listSucai({
      category: categoryFilter.value,
      page: page.value,
      pageSize: pageSize.value,
    })
    items.value = body.items
    total.value = body.total
    // deleting the last item of a later page should pull back a page
    if (!body.items.length && page.value > 1) {
      page.value -= 1
      loading.value = false
      return refresh()
    }
  } catch (e) {
    error.value = e.message
  } finally {
    loading.value = false
  }
}

function setCategoryFilter(value) {
  categoryFilter.value = value
  page.value = 1
  refresh()
}

function setPage(value) {
  page.value = value
  refresh()
}

function setPageSize(value) {
  pageSize.value = value
  page.value = 1
  refresh()
}

onMounted(() => { refresh(); refreshCategories() })

function pickFile(event) {
  setFile(event.target.files?.[0])
}

function setFile(file) {
  if (!file) return
  if (!file.type.startsWith('image/')) {
    error.value = '请选择图片文件（png / jpg / webp）'
    return
  }
  pickedFile.value = file
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = URL.createObjectURL(file)
}

function clearFile() {
  pickedFile.value = null
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = ''
}

function onDrop(event) {
  dragging.value = false
  setFile(event.dataTransfer?.files?.[0])
}

function resetForm() {
  idInput.value = ''
  describeInput.value = ''
  clearFile()
}

async function submit() {
  if (!pickedFile.value) { error.value = '请先选择素材图片'; return }
  submitting.value = true
  error.value = ''
  try {
    const created = await createSucai({
      file: pickedFile.value,
      id: idInput.value.trim() || undefined,
      describe: describeInput.value.trim(),
      category: categoryInput.value.trim(),
    })
    flashNotice(`已注册素材 ${created.id}`)
    resetForm()
    page.value = 1
    await Promise.all([refresh(), refreshCategories()])
  } catch (e) {
    error.value = e.message
  } finally {
    submitting.value = false
  }
}

async function remove(item) {
  if (!confirm(`删除素材 ${item.id} ?`)) return
  try {
    await deleteSucai(item.id)
    flashNotice(`已删除 ${item.id}`)
    await Promise.all([refresh(), refreshCategories()])
  } catch (e) {
    error.value = e.message
  }
}

function fmtSize(bytes) {
  return bytes > 1024 ? `${(bytes / 1024).toFixed(1)} KB` : `${bytes} B`
}
</script>

<template>
  <div class="library">
    <section class="card form-card" aria-label="注册素材">
      <h2>注册素材</h2>
      <form @submit.prevent="submit">
        <div
          class="dropzone"
          :class="{ filled: previewUrl, dragging }"
          @dragover.prevent="dragging = true"
          @dragleave="dragging = false"
          @drop="onDrop"
        >
          <template v-if="previewUrl">
            <img :src="previewUrl" alt="素材预览" />
            <button type="button" class="icon-btn clear ghost" title="移除图片" @click="clearFile">
              <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M18 6 6 18M6 6l12 12"/></svg>
            </button>
          </template>
          <div v-else class="hint">
            <svg viewBox="0 0 24 24" width="22" height="22" fill="none" stroke="currentColor" stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
              <rect x="3" y="3" width="18" height="18" rx="3"/>
              <circle cx="9" cy="9" r="2"/>
              <path d="m21 15-3.5-3.5L7 22"/>
            </svg>
            拖拽图片到这里，或点击选择
          </div>
          <input type="file" accept="image/*" aria-label="选择素材图片" @change="pickFile" />
        </div>

        <label class="field">
          <span>素材 ID（留空自动生成）</span>
          <input v-model="idInput" placeholder="如 btn-confirm" spellcheck="false" />
        </label>
        <label class="field">
          <span>分类 category（可留空）</span>
          <input
            v-model="categoryInput"
            list="category-options"
            placeholder="如 按钮 / 图标 / 弹窗"
            spellcheck="false"
          />
          <datalist id="category-options">
            <option v-for="c in categories" :key="c" :value="c" />
          </datalist>
        </label>
        <label class="field">
          <span>描述 describe</span>
          <textarea v-model="describeInput" placeholder="这个素材是什么、在什么界面出现…" />
        </label>
        <button type="submit" class="submit" :disabled="submitting || !pickedFile">
          <span v-if="submitting" class="spinner" aria-hidden="true" />
          {{ submitting ? '注册中…' : '注册素材' }}
        </button>
      </form>
    </section>

    <section class="list-area" aria-label="素材列表">
      <div class="toolbar">
        <select
          class="cat-filter"
          :value="categoryFilter"
          aria-label="按分类筛选"
          @change="setCategoryFilter($event.target.value)"
        >
          <option value="">全部分类</option>
          <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
        </select>
        <span class="dim">{{ total }} 个素材</span>
        <span class="spacer" />
        <label class="dim page-size">
          每页
          <select :value="pageSize" aria-label="每页数量" @change="setPageSize(Number($event.target.value))">
            <option :value="12">12</option>
            <option :value="24">24</option>
            <option :value="48">48</option>
          </select>
        </label>
      </div>

      <p v-if="error" class="msg error" role="alert">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/></svg>
        {{ error }}
      </p>
      <p v-if="notice" class="msg ok" role="status">
        <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 6 9 17l-5-5"/></svg>
        {{ notice }}
      </p>

      <!-- skeleton while loading -->
      <div v-if="loading && !items.length" class="grid" aria-hidden="true">
        <div v-for="i in 8" :key="i" class="card sucai-card skeleton">
          <div class="sk-img" />
          <div class="sk-meta">
            <div class="sk-line w40" />
            <div class="sk-line w80" />
          </div>
        </div>
      </div>

      <!-- teaching empty state -->
      <div v-else-if="!items.length" class="empty card">
        <svg class="empty-art" viewBox="0 0 96 64" width="120" aria-hidden="true">
          <rect x="14" y="8" width="68" height="48" rx="6" fill="none" stroke="var(--border-strong)" stroke-width="2" stroke-dasharray="5 5"/>
          <rect x="34" y="22" width="28" height="20" rx="3" fill="none" stroke="var(--accent)" stroke-width="2"/>
          <path d="M48 14v6M48 44v6M40 32H28M68 32H56" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" opacity=".55"/>
        </svg>
        <p class="empty-title">{{ categoryFilter ? `「${categoryFilter}」分类下还没有素材` : '还没有任何素材' }}</p>
        <ol class="empty-steps">
          <li>在左侧选择一张按钮 / 图标的小图</li>
          <li>填上 ID（可留空）和分类，点注册</li>
          <li>到「查找定位」上传截图即可秒查它在哪</li>
        </ol>
      </div>

      <transition-group v-else name="grid" tag="div" class="grid">
        <article v-for="item in items" :key="item.id" class="card sucai-card">
          <button
            class="icon-btn del danger-ghost"
            title="删除素材"
            :aria-label="`删除素材 ${item.id}`"
            @click="remove(item)"
          >
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M3 6h18M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2m3 0v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6"/></svg>
          </button>
          <a :href="sucaiImageUrl(item.id)" target="_blank" rel="noopener" :aria-label="`查看大图 ${item.id}`">
            <img class="thumb" loading="lazy" :src="sucaiImageUrl(item.id)" :alt="item.describe || item.id" />
          </a>
          <div class="meta">
            <div class="tags">
              <code class="sid" :title="item.id">{{ item.id }}</code>
              <span v-if="item.category" class="cat">{{ item.category }}</span>
            </div>
            <p class="desc" :title="item.describe">{{ item.describe || '—' }}</p>
            <span class="dims">{{ item.width }}×{{ item.height }} · {{ fmtSize(item.size_bytes) }}</span>
          </div>
        </article>
      </transition-group>

      <nav v-if="totalPages > 1" class="pager" aria-label="分页">
        <button class="ghost" :disabled="page <= 1" @click="setPage(page - 1)">
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m15 18-6-6 6-6"/></svg>
          上一页
        </button>
        <span class="dim">第 {{ page }} / {{ totalPages }} 页</span>
        <button class="ghost" :disabled="page >= totalPages" @click="setPage(page + 1)">
          下一页
          <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="m9 18 6-6-6-6"/></svg>
        </button>
      </nav>
    </section>
  </div>
</template>

<style scoped>
.library {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 22px;
  align-items: start;
}
@media (max-width: 900px) {
  .library { grid-template-columns: 1fr; }
}
.form-card h2 {
  margin: 0 0 16px;
  font-size: 16px;
  font-weight: 600;
}
.form-card .submit { width: 100%; }

/* dropzone */
.dropzone {
  position: relative;
  border: 1.5px dashed var(--border-strong);
  border-radius: 10px;
  min-height: 130px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 16px;
  overflow: hidden;
  cursor: pointer;
  background: var(--panel-2);
  transition: border-color var(--t-fast) var(--ease), background var(--t-fast) var(--ease);
}
.dropzone:hover, .dropzone.dragging { border-color: var(--accent); background: var(--accent-subtle); }
.dropzone img { max-width: 100%; max-height: 180px; object-fit: contain; }
.dropzone .hint {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
  color: var(--text-dim);
  font-size: 13px;
  padding: 16px;
  text-align: center;
  pointer-events: none;
}
.dropzone input[type=file] {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
}
.dropzone .clear {
  position: absolute;
  top: 8px;
  right: 8px;
  background: rgba(13, 15, 19, .8);
  backdrop-filter: blur(4px);
}

/* toolbar */
.toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
}
.toolbar .spacer { flex: 1; }
.dim { color: var(--text-dim); font-size: 13px; }
.cat-filter, .page-size select { width: auto; }

/* notices */
.msg { margin-bottom: 12px; }

/* empty state */
.empty {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 6px;
  padding: 28px;
}
.empty-art { margin-bottom: 8px; }
.empty-title { margin: 0; font-size: 15px; font-weight: 600; }
.empty-steps {
  margin: 6px 0 0;
  padding-left: 18px;
  color: var(--text-dim);
  font-size: 13px;
  display: grid;
  gap: 4px;
}

/* grid */
.grid {
  position: relative;
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 14px;
}
.sucai-card {
  position: relative;
  padding: 0;
  overflow: hidden;
  transition: border-color var(--t-fast) var(--ease), box-shadow var(--t-med) var(--ease),
    transform var(--t-med) var(--ease);
}
.sucai-card:hover {
  border-color: var(--border-strong);
  box-shadow: var(--shadow-2);
  transform: translateY(-2px);
}
.sucai-card a {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 130px;
  background: #fff;
}
.thumb {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  transition: transform var(--t-med) var(--ease);
}
.sucai-card:hover .thumb { transform: scale(1.04); }
.meta { padding: 10px 12px 12px; border-top: 1px solid var(--border); }
.tags { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.sid {
  font-size: 12px;
  color: var(--accent);
  word-break: break-all;
}
.cat {
  font-size: 11px;
  color: var(--accent);
  background: var(--accent-subtle);
  border-radius: 999px;
  padding: 1px 8px;
  white-space: nowrap;
}
.desc {
  margin: 6px 0;
  font-size: 13px;
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  min-height: 20px;
}
.dims { font-size: 11px; color: var(--text-faint); }
.del {
  position: absolute;
  top: 8px;
  right: 8px;
  z-index: 1;
  background: rgba(13, 15, 19, .78);
  opacity: 0;
  transition: opacity var(--t-fast) var(--ease), border-color var(--t-fast) var(--ease);
}
.sucai-card:hover .del, .del:focus-visible { opacity: 1; }

/* skeleton */
.skeleton .sk-img { height: 130px; background: var(--panel-2); }
.skeleton .sk-meta { padding: 12px; display: grid; gap: 8px; }
.sk-line { height: 10px; border-radius: 5px; background: var(--panel-2); position: relative; overflow: hidden; }
.sk-line::after {
  content: '';
  position: absolute;
  inset: 0;
  transform: translateX(-100%);
  background: linear-gradient(90deg, transparent, rgba(255, 255, 255, .06), transparent);
  animation: shimmer 1.3s infinite;
}
.w40 { width: 40%; }
.w80 { width: 80%; }
@keyframes shimmer { to { transform: translateX(100%); } }

/* pager */
.pager {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  margin-top: 18px;
}

/* transitions */
.grid-enter-active, .grid-leave-active { transition: opacity .2s var(--ease), transform .2s var(--ease); }
.grid-enter-from, .grid-leave-to { opacity: 0; transform: scale(.96); }
.grid-leave-active { position: absolute; }

@media (max-width: 640px) {
  .toolbar { flex-wrap: wrap; }
  .del { opacity: 1; } /* touch: no hover */
}
</style>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { createSucai, deleteSucai, listSucai, listSucaiCategories, sucaiImageUrl } from '../api.js'

const items = ref([])
const total = ref(0)
const categories = ref([])
const loading = ref(false)
const error = ref('')
const notice = ref('')

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
const submitting = ref(false)

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
  const file = event.target.files?.[0]
  setFile(file)
}

function setFile(file) {
  if (!file) return
  if (!file.type.startsWith('image/')) {
    error.value = '请选择图片文件'
    return
  }
  pickedFile.value = file
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = URL.createObjectURL(file)
}

function onDrop(event) {
  event.preventDefault()
  setFile(event.dataTransfer?.files?.[0])
}

function resetForm() {
  idInput.value = ''
  describeInput.value = ''
  pickedFile.value = null
  if (previewUrl.value) URL.revokeObjectURL(previewUrl.value)
  previewUrl.value = ''
}

async function submit() {
  if (!pickedFile.value) { error.value = '请先选择素材图片'; return }
  submitting.value = true
  error.value = ''
  notice.value = ''
  try {
    const created = await createSucai({
      file: pickedFile.value,
      id: idInput.value.trim() || undefined,
      describe: describeInput.value.trim(),
      category: categoryInput.value.trim(),
    })
    notice.value = `已注册素材 ${created.id}`
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
    <section class="card form-card">
      <h2>注册素材</h2>
      <form @submit.prevent="submit">
        <div
          class="dropzone"
          :class="{ filled: previewUrl }"
          @dragover.prevent
          @drop="onDrop"
        >
          <img v-if="previewUrl" :src="previewUrl" alt="预览" />
          <div v-else class="hint">拖拽图片到这里，或点击选择</div>
          <input type="file" accept="image/*" @change="pickFile" />
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
        <button type="submit" :disabled="submitting || !pickedFile">
          {{ submitting ? '注册中…' : '注册' }}
        </button>
      </form>
    </section>

    <section class="list-area">
      <div class="toolbar">
        <select class="cat-filter" :value="categoryFilter" @change="setCategoryFilter($event.target.value)">
          <option value="">全部分类</option>
          <option v-for="c in categories" :key="c" :value="c">{{ c }}</option>
        </select>
        <span class="dim">{{ total }} 个素材</span>
        <span class="spacer" />
        <label class="dim">
          每页
          <select class="size-filter" :value="pageSize" @change="setPageSize(Number($event.target.value))">
            <option :value="12">12</option>
            <option :value="24">24</option>
            <option :value="48">48</option>
          </select>
        </label>
      </div>

      <p v-if="error" class="msg error">{{ error }}</p>
      <p v-if="notice" class="msg ok">{{ notice }}</p>

      <p v-if="!loading && !items.length" class="empty">
        {{ categoryFilter ? '该分类下还没有素材。' : '还没有素材，先在左侧注册一个吧。' }}
      </p>

      <transition-group name="grid" tag="div" class="grid">
        <article v-for="item in items" :key="item.id" class="card sucai-card">
          <button class="del danger" title="删除" @click="remove(item)">✕</button>
          <a :href="sucaiImageUrl(item.id)" target="_blank" rel="noopener">
            <img class="thumb" :src="sucaiImageUrl(item.id)" :alt="item.describe || item.id" />
          </a>
          <div class="meta">
            <div class="tags">
              <code class="sid">{{ item.id }}</code>
              <span v-if="item.category" class="cat">{{ item.category }}</span>
            </div>
            <p class="desc">{{ item.describe || '—' }}</p>
            <span class="dims">{{ item.width }}×{{ item.height }} · {{ fmtSize(item.size_bytes) }}</span>
          </div>
        </article>
      </transition-group>

      <div v-if="totalPages > 1" class="pager">
        <button class="ghost" :disabled="page <= 1" @click="setPage(page - 1)">上一页</button>
        <span class="dim">第 {{ page }} / {{ totalPages }} 页</span>
        <button class="ghost" :disabled="page >= totalPages" @click="setPage(page + 1)">下一页</button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.library {
  display: grid;
  grid-template-columns: 320px 1fr;
  gap: 20px;
  align-items: start;
}
@media (max-width: 860px) {
  .library { grid-template-columns: 1fr; }
}
.card {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 16px;
}
.form-card h2 { margin: 0 0 14px; font-size: 16px; }

.dropzone {
  position: relative;
  border: 1px dashed var(--border);
  border-radius: 10px;
  min-height: 120px;
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 14px;
  overflow: hidden;
  cursor: pointer;
  background: var(--panel-2);
}
.dropzone:hover { border-color: var(--accent); }
.dropzone img { max-width: 100%; max-height: 180px; object-fit: contain; }
.dropzone .hint { color: var(--text-dim); font-size: 13px; padding: 12px; text-align: center; }
.dropzone input[type=file] {
  position: absolute;
  inset: 0;
  opacity: 0;
  cursor: pointer;
}

.msg { border-radius: 8px; padding: 8px 12px; font-size: 13px; }
.msg.error { background: rgba(255, 93, 93, .12); color: var(--bad); }
.msg.ok { background: rgba(52, 199, 123, .12); color: var(--ok); }
.empty { color: var(--text-dim); }
.dim { color: var(--text-dim); font-size: 13px; }

.toolbar {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
}
.toolbar .spacer { flex: 1; }
.cat-filter, .size-filter {
  width: auto;
  padding: 6px 10px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 8px;
  color: var(--text);
}

.pager {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 16px;
  margin-top: 18px;
}
.pager button { padding: 6px 16px; }

.grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
  gap: 14px;
}
.sucai-card { position: relative; padding: 0; overflow: hidden; }
.sucai-card a {
  display: flex;
  align-items: center;
  justify-content: center;
  height: 130px;
  background: #fff;
}
.thumb { max-width: 100%; max-height: 100%; object-fit: contain; }
.meta { padding: 10px 12px 12px; border-top: 1px solid var(--border); }
.tags { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
.sid { font-size: 12px; color: var(--accent); word-break: break-all; }
.cat {
  font-size: 11px;
  color: var(--accent);
  background: rgba(79, 140, 255, .12);
  border-radius: 999px;
  padding: 1px 8px;
  white-space: nowrap;
}
.desc {
  margin: 6px 0;
  font-size: 13px;
  color: var(--text);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
  min-height: 18px;
}
.dims { font-size: 11px; color: var(--text-dim); }
.del {
  position: absolute;
  top: 6px;
  right: 6px;
  z-index: 1;
  width: 26px;
  height: 26px;
  padding: 0;
  line-height: 1;
  border-radius: 50%;
  background: rgba(15, 17, 21, .75);
}

.grid-enter-active, .grid-leave-active { transition: all .25s ease; }
.grid-enter-from, .grid-leave-to { opacity: 0; transform: scale(.95); }
.grid-leave-active { position: absolute; }
</style>

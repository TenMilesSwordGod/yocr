const BASE = '/api/v1'

async function handle(resp) {
  if (!resp.ok) {
    let detail = `${resp.status} ${resp.statusText}`
    try {
      const body = await resp.json()
      if (body && body.detail) detail = typeof body.detail === 'string' ? body.detail : JSON.stringify(body.detail)
    } catch { /* non-json error body */ }
    throw new Error(detail)
  }
  return resp.json()
}

export function sucaiImageUrl(id) {
  return `${BASE}/sucai/${encodeURIComponent(id)}/image`
}

export async function listSucai({ category = '', page = 1, pageSize = 24 } = {}) {
  const q = new URLSearchParams({ page: String(page), page_size: String(pageSize) })
  if (category) q.set('category', category)
  return handle(await fetch(`${BASE}/sucai?${q}`))
}

export async function listSucaiCategories() {
  return handle(await fetch(`${BASE}/sucai/categories`))
}

export async function createSucai({ file, id, describe, category }) {
  const form = new FormData()
  form.append('file', file)
  if (id) form.append('id', id)
  if (describe) form.append('describe', describe)
  if (category) form.append('category', category)
  return handle(await fetch(`${BASE}/sucai`, { method: 'POST', body: form }))
}

export async function updateSucai(id, { file, describe, category } = {}) {
  const form = new FormData()
  if (file) form.append('file', file)
  if (describe !== undefined && describe !== null) form.append('describe', describe)
  if (category !== undefined && category !== null) form.append('category', category)
  return handle(await fetch(`${BASE}/sucai/${encodeURIComponent(id)}`, { method: 'PUT', body: form }))
}

export async function deleteSucai(id) {
  return handle(await fetch(`${BASE}/sucai/${encodeURIComponent(id)}`, { method: 'DELETE' }))
}

export async function findSucai({ sceneFile, threshold, topK = 0, allInstances = false }) {
  const form = new FormData()
  form.append('file', sceneFile)
  return handle(await fetch(
    `${BASE}/sucai/find?threshold=${threshold}&top_k=${topK}&all_instances=${allInstances}`,
    { method: 'POST', body: form },
  ))
}

// ------------------------------------------------ vision analysis --------
export async function listModels() {
  return handle(await fetch(`${BASE}/models`))
}

function visionQuery({ model, conf, iou, imgsz, text, label, q, matchMode } = {}) {
  const qs = new URLSearchParams()
  if (model) qs.set('model', model)
  if (conf != null) qs.set('conf', conf)
  if (iou != null) qs.set('iou', iou)
  if (imgsz) qs.set('imgsz', imgsz)
  if (text) qs.set('text', text)
  if (label) qs.set('label', label)
  if (q) qs.set('q', q)
  if (matchMode && matchMode !== 'contains') qs.set('match_mode', matchMode)
  return qs
}

export async function detect({ file, ...params }) {
  const form = new FormData()
  form.append('file', file)
  return handle(await fetch(`${BASE}/detect?${visionQuery(params)}`, { method: 'POST', body: form }))
}

export async function ocr({ file }) {
  const form = new FormData()
  form.append('file', file)
  return handle(await fetch(`${BASE}/ocr`, { method: 'POST', body: form }))
}

export async function analyze({ file, withOcr = true, ...params }) {
  const qs = visionQuery(params)
  qs.set('with_ocr', String(withOcr))
  const form = new FormData()
  form.append('file', file)
  return handle(await fetch(`${BASE}/analyze?${qs}`, { method: 'POST', body: form }))
}

export async function matchTemplate({ sceneFile, templateFile, threshold }) {
  const form = new FormData()
  form.append('file', sceneFile)
  form.append('template', templateFile)
  return handle(await fetch(`${BASE}/match?threshold=${threshold}`, { method: 'POST', body: form }))
}

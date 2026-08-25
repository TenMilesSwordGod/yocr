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

export async function listSucai() {
  return handle(await fetch(`${BASE}/sucai`))
}

export async function createSucai({ file, id, describe }) {
  const form = new FormData()
  form.append('file', file)
  if (id) form.append('id', id)
  if (describe) form.append('describe', describe)
  return handle(await fetch(`${BASE}/sucai`, { method: 'POST', body: form }))
}

export async function updateSucai(id, { file, describe } = {}) {
  const form = new FormData()
  if (file) form.append('file', file)
  if (describe !== undefined && describe !== null) form.append('describe', describe)
  return handle(await fetch(`${BASE}/sucai/${encodeURIComponent(id)}`, { method: 'PUT', body: form }))
}

export async function deleteSucai(id) {
  return handle(await fetch(`${BASE}/sucai/${encodeURIComponent(id)}`, { method: 'DELETE' }))
}

export async function findSucai({ sceneFile, threshold, topK = 0 }) {
  const form = new FormData()
  form.append('file', sceneFile)
  return handle(await fetch(
    `${BASE}/sucai/find?threshold=${threshold}&top_k=${topK}`,
    { method: 'POST', body: form },
  ))
}

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

export async function drawScene(canvas, sceneUrl) {
  const img = await loadImage(sceneUrl)
  canvas.width = img.naturalWidth
  canvas.height = img.naturalHeight
  const ctx = canvas.getContext('2d')
  ctx.drawImage(img, 0, 0)
  return ctx
}

export function strokeBox(ctx, xyxy, { color = '#ffb224', width = 2.5, dash = [], label = '' } = {}) {
  ctx.save()
  ctx.lineWidth = width
  ctx.strokeStyle = color
  ctx.setLineDash(dash)
  ctx.strokeRect(...xyxy)
  if (label) {
    ctx.font = '13px sans-serif'
    const w = ctx.measureText(label).width + 10
    const ly = Math.max(xyxy[1] - 20, 0)
    ctx.fillStyle = color
    ctx.fillRect(xyxy[0], ly, w, 19)
    ctx.fillStyle = '#10131a'
    ctx.fillText(label, xyxy[0] + 5, ly + 14)
  }
  ctx.restore()
}

export function truncate(text, max = 18) {
  const s = String(text ?? '')
  return s.length > max ? `${s.slice(0, max)}…` : s
}

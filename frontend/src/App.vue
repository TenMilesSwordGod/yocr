<script setup>
import { onMounted, ref } from 'vue'
import SucaiLibrary from './components/SucaiLibrary.vue'
import SucaiFinder from './components/SucaiFinder.vue'
import VisionLab from './components/VisionLab.vue'
import TemplateMatch from './components/TemplateMatch.vue'

const tab = ref('library')
const health = ref('checking') // checking | ok | down

async function checkHealth() {
  health.value = 'checking'
  try {
    const r = await fetch('/api/v1/healthz', { cache: 'no-store' })
    health.value = r.ok ? 'ok' : 'down'
  } catch {
    health.value = 'down'
  }
}

onMounted(checkHealth)
</script>

<template>
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <svg class="logo-mark" viewBox="0 0 24 24" width="26" height="26" aria-hidden="true">
          <rect x="3" y="3" width="18" height="18" rx="4" fill="var(--accent)" />
          <circle cx="12" cy="12" r="4.2" fill="none" stroke="#fff" stroke-width="1.8" />
          <path d="M12 5.4v3.2M12 15.4v3.2M5.4 12h3.2M15.4 12h3.2" stroke="#fff" stroke-width="1.8" stroke-linecap="round" />
        </svg>
        <div class="brand-text">
          <span class="logo">yocr</span>
          <span class="sub">素材注册与快速定位</span>
        </div>
      </div>

      <nav class="tabs" role="tablist" aria-label="功能切换">
        <button
          role="tab"
          :aria-selected="tab === 'library'"
          :class="{ active: tab === 'library' }"
          class="tab ghost"
          @click="tab = 'library'"
        >素材库</button>
        <button
          role="tab"
          :aria-selected="tab === 'finder'"
          :class="{ active: tab === 'finder' }"
          class="tab ghost"
          @click="tab = 'finder'"
        >查找定位</button>
        <button
          role="tab"
          :aria-selected="tab === 'vision'"
          :class="{ active: tab === 'vision' }"
          class="tab ghost"
          @click="tab = 'vision'"
        >视觉分析</button>
        <button
          role="tab"
          :aria-selected="tab === 'match'"
          :class="{ active: tab === 'match' }"
          class="tab ghost"
          @click="tab = 'match'"
        >模板匹配</button>
      </nav>

      <div class="health" :class="health" :title="`服务状态：${health}`">
        <span class="dot" aria-hidden="true" />
        <span class="health-label">{{ health === 'ok' ? '服务正常' : health === 'down' ? '服务异常' : '检测中…' }}</span>
      </div>
    </header>

    <main class="content">
      <div v-show="tab === 'library'" role="tabpanel" aria-label="素材库">
        <SucaiLibrary />
      </div>
      <div v-show="tab === 'finder'" role="tabpanel" aria-label="查找定位">
        <SucaiFinder />
      </div>
      <div v-show="tab === 'vision'" role="tabpanel" aria-label="视觉分析">
        <VisionLab />
      </div>
      <div v-show="tab === 'match'" role="tabpanel" aria-label="模板匹配">
        <TemplateMatch />
      </div>
    </main>
  </div>
</template>

<style scoped>
.shell {
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 24px 56px;
}
.topbar {
  position: sticky;
  top: 0;
  z-index: 10;
  display: flex;
  align-items: center;
  gap: 20px;
  padding: 12px 0;
  margin-bottom: 24px;
  background: color-mix(in srgb, var(--bg) 86%, transparent);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--border);
}
.brand { display: flex; align-items: center; gap: 10px; margin-right: auto; }
.logo-mark { display: block; }
.brand-text { display: flex; flex-direction: column; line-height: 1.25; }
.logo {
  font-size: 17px;
  font-weight: 700;
  letter-spacing: .02em;
  color: var(--text);
}
.sub { color: var(--text-dim); font-size: 12px; }

.tabs {
  display: flex;
  gap: 4px;
  background: var(--panel-2);
  border: 1px solid var(--border);
  border-radius: 999px;
  padding: 3px;
}
.tab {
  min-height: 30px;
  padding: 0 18px;
  border-radius: 999px;
  border-color: transparent;
  font-size: 13px;
}
.tab:hover:not(:disabled) { background: transparent; color: var(--text); }
.tab.active {
  background: var(--accent);
  color: #fff;
}
.tab.active:hover:not(:disabled) { background: var(--accent-hover); }

.health {
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 12px;
  color: var(--text-dim);
  flex: none;
}
.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--text-faint);
}
.health.ok .dot { background: var(--ok); box-shadow: 0 0 0 3px var(--ok-subtle); }
.health.down .dot { background: var(--bad); box-shadow: 0 0 0 3px var(--bad-subtle); }
.health.checking .dot { background: var(--warn); }

@media (max-width: 720px) {
  .shell { padding: 0 14px 40px; }
  .topbar { flex-wrap: wrap; gap: 10px; }
  .sub { display: none; }
  .health { margin-left: auto; }
  .tab { padding: 0 12px; font-size: 12px; min-height: 28px; }
}
@media (prefers-reduced-motion: reduce) {
  .topbar { backdrop-filter: none; background: var(--bg); }
}
</style>

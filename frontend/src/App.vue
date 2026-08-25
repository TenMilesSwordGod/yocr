<script setup>
import { ref } from 'vue'
import SucaiLibrary from './components/SucaiLibrary.vue'
import SucaiFinder from './components/SucaiFinder.vue'

const tab = ref('library')
</script>

<template>
  <div class="shell">
    <header class="topbar">
      <div class="brand">
        <span class="logo">yocr</span>
        <span class="sub">素材注册与快速定位</span>
      </div>
      <nav class="tabs">
        <button
          :class="{ active: tab === 'library' }"
          class="tab ghost"
          @click="tab = 'library'"
        >素材库</button>
        <button
          :class="{ active: tab === 'finder' }"
          class="tab ghost"
          @click="tab = 'finder'"
        >查找定位</button>
      </nav>
    </header>

    <main class="content">
      <SucaiLibrary v-show="tab === 'library'" />
      <SucaiFinder v-show="tab === 'finder'" />
    </main>
  </div>
</template>

<style scoped>
.shell {
  max-width: 1280px;
  margin: 0 auto;
  padding: 0 20px 40px;
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 16px 0;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
}
.brand { display: flex; align-items: baseline; gap: 10px; }
.logo {
  font-size: 22px;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: .5px;
}
.sub { color: var(--text-dim); font-size: 13px; }
.tabs { display: flex; gap: 8px; }
.tab { padding: 7px 20px; border-radius: 999px; }
.tab.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.content { min-height: 60vh; }
</style>

import './assets/main.css'

import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './App.vue'
import router from './router'
import { GlobalState } from './domain/global_state/global_state'

GlobalState.initializeDefault();
const app = createApp(App)

app.use(createPinia())
app.use(router)

app.mount('#app')

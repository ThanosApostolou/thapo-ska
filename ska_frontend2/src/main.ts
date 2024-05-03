import './assets/main.css'

import { createApp } from 'vue'
import { createPinia } from 'pinia'

import App from './gui/App.vue'
import { GlobalState } from './domain/global_state/global_state'
import { myCreateRouter } from './gui/router'

const globalState = GlobalState.initializeDefault();
const app = createApp(App)

app.use(createPinia())
app.use(myCreateRouter(globalState))

app.mount('#app')

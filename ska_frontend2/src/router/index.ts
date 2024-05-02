import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../gui/page_home/PageView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },
    {
      path: '/about',
      name: 'about',
      // route level code-splitting
      // this generates a separate chunk (About.[hash].js) for this route
      // which is lazy-loaded when the route is visited.
      component: () => import('../gui/page_about/PageAbout.vue')
    }
  ]
})

export default router

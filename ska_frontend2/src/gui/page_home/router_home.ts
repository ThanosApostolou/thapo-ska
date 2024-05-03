import type { RouteRecordRaw } from 'vue-router';

export function createHomeRoutes(): RouteRecordRaw {
    const routes = {
        path: '/home',
        name: 'home',
        component: () => import('./PageHome.vue')
    };
    return routes;
}
import type { RouteRecordRaw } from 'vue-router';

export function createAboutRoutes(): RouteRecordRaw {
    const routes = {
        path: '/about',
        name: 'about',
        component: () => import('./PageAbout.vue')
    };
    return routes;
}
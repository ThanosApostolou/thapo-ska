import type { AppRoutes } from '@/domain/global_state/app_routes';
import type { RouteRecordRaw } from 'vue-router';

export function createRouterHome(appRoutes: AppRoutes): RouteRecordRaw {
    const routes = {
        path: appRoutes.PAGE_HOME,
        name: 'home',
        component: () => import('./PageHome.vue')
    };
    return routes;
}
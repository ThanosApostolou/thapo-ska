import type { AppRoutes } from '@/domain/global_state/app_routes';
import type { RouteRecordRaw } from 'vue-router';

export function createRouterLogin(appRoutes: AppRoutes): RouteRecordRaw {
    const routes = {
        path: appRoutes.PAGE_LOGIN,
        name: 'login',
        component: () => import('./PageLogin.vue')
    };
    return routes;
}
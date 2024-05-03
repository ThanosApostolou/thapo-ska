import type { AppRoutes } from '@/domain/global_state/app_routes';
import type { RouteRecordRaw } from 'vue-router';

export function createRouterAccount(appRoutes: AppRoutes): RouteRecordRaw {
    const routes = {
        path: appRoutes.PAGE_ACCOUNT,
        name: 'account',
        component: () => import('./PageAccount.vue')
    };
    return routes;
}
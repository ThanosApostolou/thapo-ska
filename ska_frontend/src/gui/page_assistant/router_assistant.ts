import type { AppRoutes } from '@/domain/global_state/app_routes';
import type { RouteRecordRaw } from 'vue-router';

export function createRouterAssistant(appRoutes: AppRoutes): RouteRecordRaw {
    const routes = {
        path: appRoutes.PAGE_ASSISTANT,
        name: 'assistant',
        component: () => import('./PageAssistant.vue'),
        meta: { requiresAuth: true }
    };
    return routes;
}
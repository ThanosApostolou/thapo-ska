import { createRouter, createWebHistory, createMemoryHistory, type RouterHistory, type Router } from 'vue-router';
import { createRouterHome } from './page_home/router_home';
import { createRouterAccount } from './page_account/router_account';
import { GlobalState } from '@/domain/global_state/global_state';
import { BuildEnv } from '@/domain/global_state/build_env';
import { createRouterAssistant } from './page_assistant/router_assistant';
import { useGlobalStore } from '@/domain/global_state/global_store';
// import { createRouterLogin } from './page_login/router_login';

export function myCreateRouter(global_state: GlobalState): Router {
    const envConfig = global_state.envConfig;

    const history: RouterHistory = envConfig.buildEnv === BuildEnv.web ? createWebHistory(envConfig.baseUrl) : createMemoryHistory(envConfig.baseUrl);
    const appRoutes = global_state.appRoutes;
    const router = createRouter({
        history: history,
        routes: [
            {
                path: '',
                redirect: appRoutes.PAGE_HOME,
            },
            createRouterHome(appRoutes),
            createRouterAccount(appRoutes),
            createRouterAssistant(appRoutes),
            {
                path: '/:pathMatch(.*)*',
                name: 'not_found',
                component: () => import('./page_not_found/PageNotFound.vue')
            },
        ]
    });
    router.beforeEach(async (to, from, next) => {
        const globalStore = useGlobalStore();
        await globalStore.waitAppReady();
        if (to.meta.requiresAuth === true) {
            if (globalStore.globalStore.userDetails != null) {
                next();
            } else {
                next({ path: appRoutes.PAGE_HOME });
            }
        } else {
            next();
        }
    });
    return router;
}

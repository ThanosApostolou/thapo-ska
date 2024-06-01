import { createRouter, createWebHistory, createMemoryHistory, type RouterHistory } from 'vue-router';
import { createRouterHome } from './page_home/router_home';
import { createRouterAccount } from './page_acount/router_account';
import { GlobalState } from '@/domain/global_state/global_state';
import { BuildEnv } from '@/domain/global_state/build_env';
import { createRouterAssistant } from './page_assistant/router_assistant';
// import { createRouterLogin } from './page_login/router_login';

export function myCreateRouter(global_state: GlobalState) {
    const envConfig = global_state.envConfig;

    const history: RouterHistory = envConfig.buildEnv === BuildEnv.web ? createWebHistory(envConfig.baseUrl) : createMemoryHistory(envConfig.baseUrl);
    const appRoutes = global_state.appRoutes;
    return createRouter({
        history: history,
        routes: [
            {
                path: '',
                redirect: appRoutes.PAGE_HOME,
            },
            createRouterHome(appRoutes),
            createRouterAccount(appRoutes),
            createRouterAssistant(appRoutes),
            // createRouterLogin(appRoutes),
            {
                path: '/:pathMatch(.*)*',
                name: 'not_found',
                component: () => import('./page_not_found/PageNotFound.vue')
            },
        ]
    });
}

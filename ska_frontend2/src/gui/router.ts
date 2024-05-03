import { createRouter, createWebHistory, createMemoryHistory, type RouterHistory } from 'vue-router';
import { createHomeRoutes } from './page_home/router_home';
import { createAboutRoutes } from './page_about/router_about';
import { GlobalState } from '@/domain/global_state/global_state';
import { BuildEnv } from '@/domain/global_state/build_env';

export function myCreateRouter(global_state: GlobalState) {
    const envConfig = global_state.envConfig;

    const history: RouterHistory = envConfig.buildEnv === BuildEnv.web ? createWebHistory(envConfig.baseUrl) : createMemoryHistory(envConfig.baseUrl);
    return createRouter({
        history: history,
        routes: [
            {
                path: '',
                redirect: '/home',
            },
            createHomeRoutes(),
            createAboutRoutes(),
        ]
    });
}

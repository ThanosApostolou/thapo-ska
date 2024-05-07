import { User, UserManager, type UserManagerSettings } from "oidc-client-ts";
import type { EnvConfig } from "../global_state/env_config";
import { GlobalState } from "../global_state/global_state";
import type { Router } from "vue-router";
import type { Store } from "pinia";
import type { GlobalStore } from "../global_state/global_store";
import { DtoUserDetails } from "./dto_user_details";
import { Ok, type Result } from "@/utils/core/result";
import type { DtoErrorResponse } from "@/utils/error/errors";
import { UtilsHttp } from "@/utils/http/utils_http";

export class ServiceAuth {
    static readonly PATH_API_AUTH: string = "api/auth";

    static async getUser(): Promise<User | null> {
        const userManager = GlobalState.instance().userManager;
        return userManager.getUser();
    }

    static async login(): Promise<void> {
        const userManager = GlobalState.instance().userManager;
        return userManager.signinRedirect();
    }

    static async signinCallback(): Promise<void | User> {
        const userManager = GlobalState.instance().userManager;
        return userManager.signinCallback();
    }

    static async storeUser(user: User | null) {
        const userManager = GlobalState.instance().userManager;
        await userManager.storeUser(user);
    }

    static async renewToken(): Promise<User | null> {
        const userManager = GlobalState.instance().userManager;
        return userManager.signinSilent();
    }

    static async logout(): Promise<void> {
        const userManager = GlobalState.instance().userManager;
        await userManager.signoutRedirect();
    }


    static async afterLogin(router: Router): Promise<void> {
        const globalState = GlobalState.instance();
        const currentUser = await ServiceAuth.getUser();
        if (currentUser == null) {
            const user = await ServiceAuth.signinCallback();
            console.log('user', user)
            if (user != null) {
                await ServiceAuth.storeUser(user);
            }
            await router.push({ path: globalState.appRoutes.PAGE_HOME, replace: true });
            await router.go(0);
        } else {
            await router.push({ path: globalState.appRoutes.PAGE_HOME, replace: true });
        }

    }


    static async initialAuth(globalStore: GlobalStore): Promise<void> {
        const globalState = GlobalState.instance();
        console.log('globalState', globalState)
        const user = await ServiceAuth.getUser();
        if (user != null) {
            const dtoUserDetailsResult = await this.app_login();
            const dtoUserDetails = dtoUserDetailsResult.unwrap();
            globalStore.userDetails = dtoUserDetails;

            // globalStore.userDetails = new DtoUserDetails(
            // globalStore

            // globalStore.idToken = user.id_token ? user.id_token : null;
            // globalStore.accessToken = user.access_token ? user.access_token : null;
            // globalStore.refreshToken = user.refresh_token ? user.refresh_token : null;
        }

    }

    static async app_login(): Promise<Result<DtoUserDetails, DtoErrorResponse>> {
        const globalState = GlobalState.instance();
        const urlAppLogin = `${globalState.envConfig.backendUrl}${this.PATH_API_AUTH}"/app_login`;

        return await UtilsHttp.getRequest<DtoUserDetails>(urlAppLogin);
    }

}

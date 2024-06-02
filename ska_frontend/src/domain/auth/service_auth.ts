import { User } from "oidc-client-ts";
import { GlobalState } from "../global_state/global_state";
import type { Router } from "vue-router";
import type { GlobalStore } from "../global_state/global_store";
import { DtoUserDetails } from "./dto_user_details";
import { type Result } from "@/utils/core/result";
import type { DtoErrorResponse } from "@/utils/error/errors";
import { UtilsHttp } from "@/utils/http/utils_http";

export class ServiceAuth {
    private static readonly PATH_API_AUTH: string = "api/auth";

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


    static async initialAuth(globalStore: GlobalStore, router: Router): Promise<void> {
        const globalState = GlobalState.instance();
        let user = await ServiceAuth.getUser();
        if (user != null) {
            user = await ServiceAuth.renewToken();
            const dtoUserDetailsResult = await this.app_login();
            const dtoUserDetails = dtoUserDetailsResult.unwrap();
            globalStore.userDetails = dtoUserDetails;
            globalStore.isAppReady = true;
        } else {
            try {
                user = await ServiceAuth.signinCallback() || null;
            } catch (e) {
                // ignore
            }
            if (user != null) {
                await ServiceAuth.storeUser(user);
                const dtoUserDetailsResult = await this.app_login();
                const dtoUserDetails = dtoUserDetailsResult.unwrap();
                globalStore.userDetails = dtoUserDetails;
                globalStore.isAppReady = true;
                await router.push({ path: globalState.appRoutes.PAGE_HOME, replace: true });
            } else {
                globalStore.isAppReady = true;
            }
        }
    }

    static async app_login(): Promise<Result<DtoUserDetails, DtoErrorResponse>> {
        const globalState = GlobalState.instance();
        const urlAppLogin = `${globalState.envConfig.backendUrl}${this.PATH_API_AUTH}/app_login`;

        const result = await UtilsHttp.postRequest<unknown>(urlAppLogin);
        return result.map((data) => DtoUserDetails.fromUnknown(data));
    }

}

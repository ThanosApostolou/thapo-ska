import { EnvConfig } from "./env_config";
import { AppRoutes } from "./app_routes";
import type { UserManager } from "oidc-client-ts";
import { createUserManager } from "../auth/create_user_manager";
import type { AxiosInstance } from "axios";
import { createHttpClient } from "./create_http_client";

export class GlobalState {
    private static _instance: GlobalState | null = null;

    readonly envConfig: EnvConfig;
    readonly appRoutes: AppRoutes;
    readonly userManager: UserManager;
    readonly httpClient: AxiosInstance;

    constructor(obj: {
        envConfig: EnvConfig,
        appRoutes: AppRoutes,
        userManager: UserManager,
        httpClient: AxiosInstance,
    }) {
        this.envConfig = obj.envConfig;
        this.appRoutes = obj.appRoutes;
        this.userManager = obj.userManager;
        this.httpClient = obj.httpClient;
    }

    static instance(): GlobalState {
        if (this._instance == null) {
            throw new Error("GlobalState instance is not initialized")
        } else {
            return this._instance;
        }
    }

    static initializeDefault(): GlobalState {
        if (this._instance != null) {
            throw new Error("GlobalState instance has already been initialized")
        }
        const envConfig = EnvConfig.fromEnv(import.meta.env);
        const appRoutes = new AppRoutes();
        const userManager = createUserManager(envConfig);
        const httpClient = createHttpClient(envConfig);
        this._instance = new GlobalState({
            envConfig,
            appRoutes,
            userManager,
            httpClient
        })
        return this._instance;
    }
}
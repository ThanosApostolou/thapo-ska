import { EnvConfig } from "./env_config";
import { AppRoutes } from "./app_routes";

export class GlobalState {
    private static _instance: GlobalState | null = null;

    readonly envConfig: EnvConfig;
    readonly appRoutes: AppRoutes;

    constructor(obj: {
        envConfig: EnvConfig,
        appRoutes: AppRoutes
    }) {
        this.envConfig = obj.envConfig;
        this.appRoutes = obj.appRoutes;
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
        this._instance = new GlobalState({
            envConfig,
            appRoutes,
        })
        return this._instance;
    }
}
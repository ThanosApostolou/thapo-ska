import { EnvConfig } from "./env_config";

export class GlobalState {
    private static _instance: GlobalState | null = null;

    readonly envConfig: EnvConfig;

    constructor(obj: {
        envConfig: EnvConfig
    }) {
        this.envConfig = obj.envConfig;
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
        this._instance = new GlobalState({
            envConfig
        })
        return this._instance;
    }
}
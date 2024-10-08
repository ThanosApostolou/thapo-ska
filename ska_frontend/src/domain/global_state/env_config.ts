import { BuildEnv, buildEnvFromString } from "./build_env";

export class EnvConfig {
    readonly envProfile: string;
    readonly buildEnv: BuildEnv;
    readonly baseUrl: string;
    readonly frontendUrl: string;
    readonly backendUrl: string;
    readonly authIssuerUrl: string;
    readonly authClientId: string;

    constructor(obj: {
        envProfile: string,
        buildEnv: BuildEnv,
        baseUrl: string,
        frontendUrl: string,
        backendUrl: string,
        authIssuerUrl: string,
        authClientId: string
    }) {
        this.envProfile = obj.envProfile;
        this.buildEnv = obj.buildEnv;
        this.baseUrl = obj.baseUrl;
        this.frontendUrl = obj.frontendUrl;
        this.backendUrl = obj.backendUrl;
        this.authIssuerUrl = obj.authIssuerUrl;
        this.authClientId = obj.authClientId;

    }

    static fromEnv(importMetaEnv: ImportMetaEnv): EnvConfig {
        const envProfile: string = importMetaEnv.VITE_THAPO_SKA_ENV_PROFILE;
        const buildEnv: BuildEnv = buildEnvFromString(importMetaEnv.VITE_THAPO_SKA_BUILD_ENV).unwrap();
        const baseUrl: string = importMetaEnv.BASE_URL;
        const frontendUrl: string = importMetaEnv.VITE_THAPO_SKA_FRONTEND_URL;
        const backendUrl: string = importMetaEnv.VITE_THAPO_SKA_BACKEND_URL;
        const authIssuerUrl: string = importMetaEnv.VITE_THAPO_SKA_AUTH_ISSUER_URL;
        const authClientId: string = importMetaEnv.VITE_THAPO_SKA_AUTH_CLIENT_ID;
        return new EnvConfig({
            envProfile,
            buildEnv,
            baseUrl,
            frontendUrl,
            backendUrl,
            authIssuerUrl,
            authClientId
        });

    }
}
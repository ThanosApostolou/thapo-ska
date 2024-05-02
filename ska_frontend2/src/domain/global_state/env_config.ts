export class EnvConfig {
    readonly envProfile: string;
    readonly baseUrl: string;
    readonly frontendUrl: string;
    readonly backendUrl: string;
    readonly authIssuerUrl: string;
    readonly authClientId: string;

    constructor(obj: {
        envProfile: string;
        baseUrl: string;
        frontendUrl: string;
        backendUrl: string;
        authIssuerUrl: string;
        authClientId: string;
    }) {
        this.envProfile = obj.envProfile;
        this.baseUrl = obj.baseUrl;
        this.frontendUrl = obj.frontendUrl;
        this.backendUrl = obj.backendUrl;
        this.authIssuerUrl = obj.authIssuerUrl;
        this.authClientId = obj.authClientId;
    }

    static fromEnv(importMetaEnv: ImportMetaEnv): EnvConfig {
        const envProfile: string = importMetaEnv.VITE_THAPO_SKA_ENV_PROFILE;
        const baseUrl: string = importMetaEnv.BASE_URL;
        const frontendUrl: string = importMetaEnv.VITE_THAPO_SKA_FRONTEND_URL;
        const backendUrl: string = importMetaEnv.VITE_THAPO_SKA_BACKEND_URL;
        const authIssuerUrl: string = importMetaEnv.VITE_THAPO_SKA_AUTH_ISSUER_URL;
        const authClientId: string = importMetaEnv.VITE_THAPO_SKA_AUTH_CLIENT_ID;
        return new EnvConfig({
            envProfile,
            baseUrl,
            frontendUrl,
            backendUrl,
            authIssuerUrl,
            authClientId
        });

    }
}

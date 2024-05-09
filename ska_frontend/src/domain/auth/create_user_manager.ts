import { UserManager, type UserManagerSettings } from "oidc-client-ts";
import type { EnvConfig } from "../global_state/env_config";

export function createUserManager(envConfig: EnvConfig): UserManager {

    const settings: UserManagerSettings = {
        authority: `${envConfig.authIssuerUrl}`,
        client_id: envConfig.authClientId,
        redirect_uri: `${envConfig.frontendUrl}login`,
        silent_redirect_uri: `${envConfig.frontendUrl}`,
        post_logout_redirect_uri: `${envConfig.frontendUrl}`,
        scope: 'openid',
        automaticSilentRenew: true,
        loadUserInfo: true
    };
    const userManager = new UserManager(settings);
    return userManager;
}
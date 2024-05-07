import { User, UserManager } from "oidc-client-ts";

export class UtilsAuth {

    static getUser(userManager: UserManager): Promise<User | null> {
        return userManager.getUser();
    }

    static login(userManager: UserManager): Promise<void> {
        return userManager.signinRedirect();
    }

    static signinCallback(userManager: UserManager): Promise<void | User> {
        return userManager.signinCallback();
    }

    static renewToken(userManager: UserManager): Promise<User | null> {
        return userManager.signinSilent();
    }

    static logout(userManager: UserManager): Promise<void> {
        return userManager.signoutRedirect();
    }

}
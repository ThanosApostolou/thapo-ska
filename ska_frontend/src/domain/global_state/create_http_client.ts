import axios, { type InternalAxiosRequestConfig } from "axios";
import type { EnvConfig } from "./env_config";
import { ServiceAuth } from "../auth/service_auth";

export function createHttpClient(envConfig: EnvConfig) {
    const httpClient = axios.create({
        baseURL: envConfig.backendUrl,
        timeout: 210000,
    });

    httpClient.interceptors.request.use(requestIntereptor)
    return httpClient;
}

async function requestIntereptor(config: InternalAxiosRequestConfig<any>) {
    // Do something before request is sent
    let user = await ServiceAuth.getUser();
    if (user?.expired) {
        user = await ServiceAuth.renewToken();
    }
    if (user?.access_token != null) {
        config.headers.Authorization = `Bearer ${user.access_token}`;
    }
    return config;
}

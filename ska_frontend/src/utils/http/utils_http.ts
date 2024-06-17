import { GlobalState } from "@/domain/global_state/global_state";
import { Err, Ok, type Result } from "../core/result";
import { DtoErrorResponse } from "../error/errors";
import type { AxiosRequestConfig } from "axios";

export class UtilsHttp {

    static async getRequest<R, D = unknown>(url: string, config?: AxiosRequestConfig<D>): Promise<Result<R, DtoErrorResponse>> {
        const httpClient = GlobalState.instance().httpClient;
        try {
            const response = await httpClient.get<R>(url, config);
            return Ok.new(response.data);
        } catch (error: any) {
            return this.handleError<R>(error);
        }

    }

    static async postRequest<R, D = unknown>(url: string, data?: D, config?: AxiosRequestConfig<D>): Promise<Result<R, DtoErrorResponse>> {
        const httpClient = GlobalState.instance().httpClient;
        try {
            const response = await httpClient.post<R>(url, data, config);
            return Ok.new(response.data);
        } catch (error: any) {
            return this.handleError<R>(error);
        }

    }

    static async putRequest<R, D = unknown>(url: string, data?: D, config?: AxiosRequestConfig<D>): Promise<Result<R, DtoErrorResponse>> {
        const httpClient = GlobalState.instance().httpClient;
        try {
            const response = await httpClient.put<R>(url, data, config);
            return Ok.new(response.data);
        } catch (error: any) {
            return this.handleError<R>(error);
        }

    }

    static async deleteRequest<R, D = unknown>(url: string, config?: AxiosRequestConfig<D>): Promise<Result<R, DtoErrorResponse>> {
        const httpClient = GlobalState.instance().httpClient;
        try {
            const response = await httpClient.delete<R>(url, config);
            return Ok.new(response.data);
        } catch (error: any) {
            return this.handleError<R>(error);
        }

    }

    private static handleError<R>(error: any): Err<R, DtoErrorResponse> {
        if (error.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx
            console.error('axios error.response', error.response);
            if (error.response.status == 422) {
                return Err.new(new DtoErrorResponse({
                    status_code: error.response.status,
                    is_unexpected_error: true,
                    packets: error.response.data.packets,
                }));

            } else {
                return Err.new(new DtoErrorResponse({
                    status_code: error.response.status,
                    is_unexpected_error: true,
                    packets: [],
                }))
            }
        } else if (error.request) {
            // The request was made but no response was received
            // `error.request` is an instance of XMLHttpRequest in the browser and an instance of
            // http.ClientRequest in node.js
            console.error('axios error.request', error.request);
            return Err.new(new DtoErrorResponse({
                status_code: 500,
                is_unexpected_error: true,
                packets: [],
            }));
        } else {
            // Something happened in setting up the request that triggered an Error
            console.error('axios error.message', error.message);
            return Err.new(new DtoErrorResponse({
                status_code: 500,
                is_unexpected_error: true,
                packets: [],
            }));
        }

    }
}
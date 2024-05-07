import { GlobalState } from "@/domain/global_state/global_state";
import { Err, Ok, type Result } from "../core/result";
import { DtoErrorResponse } from "../error/errors";
import type { AxiosRequestConfig } from "axios";

export class UtilsHttp {

    static async getRequest<O>(url: string, config?: AxiosRequestConfig<any>): Promise<Result<O, DtoErrorResponse>> {
        const httpClient = GlobalState.instance().httpClient;
        try {
            const response = await httpClient.get<O>(url, config);
            return Ok.new(response.data);
        } catch (error: any) {
            return this.handleError<O>(error);
        }

    }

    private static handleError<O>(error: any): Err<O, DtoErrorResponse> {
        if (error.response) {
            // The request was made and the server responded with a status code
            // that falls out of the range of 2xx
            console.error('axios error.response', error.response);
            if (error.response.status == 422) {
                // TODO
                return Err.new(new DtoErrorResponse({
                    status_code: 500,
                    is_unexpected_error: true,
                    packets: [],
                }));

            } else {
                return Err.new(new DtoErrorResponse({
                    status_code: 500,
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
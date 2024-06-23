import { GlobalState } from "@/domain/global_state/global_state";
import type { Result } from "@/utils/core/result";
import type { DtoErrorResponse } from "@/utils/error/errors";
import { UtilsHttp } from "@/utils/http/utils_http";
import { AskAssistantQuestionRequest, AskAssistantQuestionResponse } from "./dtos/dto_ask_assistant_question";
import { DtoAssistantOptions } from "./dtos/dto_fetch_assistant_options";
import { DtoCreateUpdateChatResponse, type DtoChatDetails } from "./dtos/dto_chat_details";
import { DtoFetchChatMessagesResponse, type DtoFetchChatMessagesRequest } from "./dtos/dto_fetch_chat_messages";

export class ServiceAssistant {
    private static readonly PATH_API_ASSISTANT: string = "api/assistant";


    static async askAssistantQuestion(request: AskAssistantQuestionRequest): Promise<Result<AskAssistantQuestionResponse, DtoErrorResponse>> {
        console.log('request', request)
        const globalState = GlobalState.instance();
        const url = `${globalState.envConfig.backendUrl}${this.PATH_API_ASSISTANT}/ask_assistant_question`;

        const result = await UtilsHttp.postRequest<unknown>(url, request);
        return result.map((data) => AskAssistantQuestionResponse.fromUnknown(data));
    }

    static async fetchAssistantOptions(): Promise<Result<DtoAssistantOptions, DtoErrorResponse>> {
        const globalState = GlobalState.instance();
        const url = `${globalState.envConfig.backendUrl}${this.PATH_API_ASSISTANT}/fetch_assistant_options`;

        const result = await UtilsHttp.getRequest<unknown>(url);
        return result.map((data) => DtoAssistantOptions.fromUnknown(data));
    }

    static async createChat(chatDetails: DtoChatDetails): Promise<Result<DtoCreateUpdateChatResponse, DtoErrorResponse>> {
        const globalState = GlobalState.instance();
        const url = `${globalState.envConfig.backendUrl}${this.PATH_API_ASSISTANT}/create_chat`;

        const result = await UtilsHttp.postRequest<unknown>(url, chatDetails);
        return result.map((data) => DtoCreateUpdateChatResponse.fromUnknown(data));
    }

    static async updateChat(chatDetails: DtoChatDetails): Promise<Result<DtoCreateUpdateChatResponse, DtoErrorResponse>> {
        const globalState = GlobalState.instance();
        const url = `${globalState.envConfig.backendUrl}${this.PATH_API_ASSISTANT}/update_chat`;

        const result = await UtilsHttp.putRequest<unknown>(url, chatDetails);
        return result.map((data) => DtoCreateUpdateChatResponse.fromUnknown(data));
    }

    static async deleteChat(chat_id: number): Promise<Result<DtoCreateUpdateChatResponse, DtoErrorResponse>> {
        const globalState = GlobalState.instance();
        const url = `${globalState.envConfig.backendUrl}${this.PATH_API_ASSISTANT}/delete_chat`;

        const result = await UtilsHttp.deleteRequest<unknown>(url, {
            params: {
                chat_id: chat_id
            }
        });
        return result.map((data) => DtoCreateUpdateChatResponse.fromUnknown(data));
    }

    static async fetchChatMessages(request: DtoFetchChatMessagesRequest): Promise<Result<DtoFetchChatMessagesResponse, DtoErrorResponse>> {
        const globalState = GlobalState.instance();
        const url = `${globalState.envConfig.backendUrl}${this.PATH_API_ASSISTANT}/fetch_chat_messages`;

        const result = await UtilsHttp.getRequest<unknown>(url, { params: request });
        return result.map((data) => DtoFetchChatMessagesResponse.fromUnknown(data));
    }
}
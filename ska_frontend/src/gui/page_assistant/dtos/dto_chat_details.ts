import { UtilsTypes } from "@/utils/core/utils_types";

export class DtoChatDetails {
    chat_id: number | null;
    user_id: number;
    chat_name: string;
    llm_model: string;
    prompt_template: string | null;
    temperature: number | null;
    top_p: number | null;
    default_prompt: string;

    constructor(obj: {
        chat_id: number | null,
        user_id: number,
        chat_name: string,
        llm_model: string,
        prompt_template: string | null,
        temperature: number | null,
        top_p: number | null,
        default_prompt: string,
    }) {
        this.chat_id = obj.chat_id;
        this.user_id = obj.user_id;
        this.chat_name = obj.chat_name;
        this.llm_model = obj.llm_model;
        this.prompt_template = obj.prompt_template;
        this.temperature = obj.temperature;
        this.top_p = obj.top_p;
        this.default_prompt = obj.default_prompt;
    }

    static fromUnknown(value: unknown): DtoChatDetails {
        const obj = UtilsTypes.unknownToObject(value).unwrap();
        return new DtoChatDetails({
            chat_id: UtilsTypes.unknownToNumberNullable(obj.chat_id).unwrap(),
            user_id: UtilsTypes.unknownToNumber(obj.user_id).unwrap(),
            chat_name: UtilsTypes.unknownToString(obj.chat_name).unwrap(),
            llm_model: UtilsTypes.unknownToString(obj.llm_model).unwrap(),
            prompt_template: UtilsTypes.unknownToStringNullable(obj.prompt_template).unwrap(),
            temperature: UtilsTypes.unknownToNumberNullable(obj.temperature).unwrap(),
            top_p: UtilsTypes.unknownToNumberNullable(obj.top_p).unwrap(),
            default_prompt: UtilsTypes.unknownToString(obj.default_prompt).unwrap(),
        })
    }
}


export class DtoCreateUpdateChatResponse {
    chat_id: number;

    constructor(obj: {
        chat_id: number,
    }) {
        this.chat_id = obj.chat_id;
    }

    static fromUnknown(value: unknown): DtoCreateUpdateChatResponse {
        const obj = UtilsTypes.unknownToObject(value).unwrap();
        return new DtoCreateUpdateChatResponse({
            chat_id: UtilsTypes.unknownToNumber(obj.chat_id).unwrap()
        })
    }
}

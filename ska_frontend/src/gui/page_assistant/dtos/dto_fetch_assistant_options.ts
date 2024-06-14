import { UtilsTypes } from "@/utils/core/utils_types";
import { DtoChatDetails } from "./dto_chat_details";

export class DtoAssistantOptions {
    llms: DtoLlmData[];
    user_chats: DtoChatDetails[];

    constructor(obj: {
        llms: DtoLlmData[],
        user_chats: DtoChatDetails[],
    }) {
        this.llms = obj.llms;
        this.user_chats = obj.user_chats;
    }

    static fromUnknown(value: unknown): DtoAssistantOptions {
        const obj = UtilsTypes.unknownToObject(value).unwrap();
        const llmsUnknown = UtilsTypes.unknownToArray(obj.llms).unwrap();
        const userChatsUnknown = UtilsTypes.unknownToArray(obj.user_chats).unwrap();
        return new DtoAssistantOptions({
            llms: llmsUnknown.map(DtoLlmData.fromUnknown),
            user_chats: userChatsUnknown.map(DtoChatDetails.fromUnknown)
        })
    }
}

export class DtoLlmData {
    name: string;
    default_prompt: string;

    constructor(obj: {
        name: string,
        default_prompt: string,
    }) {
        this.name = obj.name;
        this.default_prompt = obj.default_prompt;
    }

    static fromUnknown(value: unknown): DtoLlmData {
        const obj = UtilsTypes.unknownToObject(value).unwrap();
        return new DtoLlmData({
            name: UtilsTypes.unknownToString(obj.name).unwrap(),
            default_prompt: UtilsTypes.unknownToString(obj.default_prompt).unwrap()
        })
    }
}

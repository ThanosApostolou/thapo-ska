import { UtilsTypes } from "@/utils/core/utils_types";
import { DtoChatDetails } from "./dto_chat_details";
import { DtoChatPacket } from "./dto_chat_packet";

export class DtoFetchChatMessagesRequest {
    chat_id: number;

    constructor(obj: {
        chat_id: number,
    }) {
        this.chat_id = obj.chat_id;
    }
}

export class DtoFetchChatMessagesResponse {
    user_chat: DtoChatDetails;
    chat_packets: DtoChatPacket[];

    constructor(obj: {
        user_chat: DtoChatDetails,
        chat_packets: DtoChatPacket[],
    }) {
        this.user_chat = obj.user_chat;
        this.chat_packets = obj.chat_packets;
    }

    static fromUnknown(value: unknown): DtoFetchChatMessagesResponse {
        const obj = UtilsTypes.unknownToObject(value).unwrap();
        const chatPacketsUnknown = UtilsTypes.unknownToArray(obj.chat_packets).unwrap();
        return new DtoFetchChatMessagesResponse({
            user_chat: DtoChatDetails.fromUnknown(obj.user_chat),
            chat_packets: chatPacketsUnknown.map(DtoChatPacket.fromUnknown),
        })
    }
}
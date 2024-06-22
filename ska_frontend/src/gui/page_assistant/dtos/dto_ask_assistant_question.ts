import { UtilsTypes } from "@/utils/core/utils_types";
import { DtoChatPacket } from "./dto_chat_packet";

export class AskAssistantQuestionRequest {
    chat_id: number;
    question: string;

    constructor(obj: {
        chat_id: number,
        question: string,
    }) {
        this.chat_id = obj.chat_id;
        this.question = obj.question;
    }
}

export class AskAssistantQuestionResponse {
    question: DtoChatPacket;
    answer: DtoChatPacket;

    constructor(obj: {
        question: DtoChatPacket;
        answer: DtoChatPacket;
    }) {
        this.question = obj.question;
        this.answer = obj.answer;
    }

    static fromUnknown(value: unknown): AskAssistantQuestionResponse {
        const obj = UtilsTypes.unknownToObject(value).unwrap();
        return new AskAssistantQuestionResponse({
            question: DtoChatPacket.fromUnknown(obj.question),
            answer: DtoChatPacket.fromUnknown(obj.answer)
        })
    }
}

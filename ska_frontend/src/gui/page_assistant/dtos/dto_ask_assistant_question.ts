import { UtilsTypes } from "@/utils/core/utils_types";

export class AskAssistantQuestionRequest {
    question: string;
    llm_model: string;
    prompt_template: string | null;

    constructor(obj: {
        question: string,
        llm_model: string,
        prompt_template: string | null,
    }) {
        this.question = obj.question;
        this.llm_model = obj.llm_model;
        this.prompt_template = obj.prompt_template;
    }
}

export class AskAssistantQuestionResponse {
    answer: string;

    constructor(obj: {
        answer: string
    }) {
        this.answer = obj.answer;
    }

    static fromUnknown(value: unknown): AskAssistantQuestionResponse {
        const obj = UtilsTypes.unknownToObject(value).unwrap();
        return new AskAssistantQuestionResponse({
            answer: UtilsTypes.unknownToString(obj.answer).unwrap()
        })
    }
}

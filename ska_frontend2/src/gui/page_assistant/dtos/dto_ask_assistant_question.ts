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
}

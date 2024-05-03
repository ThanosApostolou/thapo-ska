export class DtoAssistantOptions {
    llms: DtoLlmData[];

    constructor(obj: {
        llms: DtoLlmData[]
    }) {
        this.llms = obj.llms;
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
}

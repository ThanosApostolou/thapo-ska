import { UtilsTypes } from "@/utils/core/utils_types";

export class DtoAssistantOptions {
    llms: DtoLlmData[];

    constructor(obj: {
        llms: DtoLlmData[]
    }) {
        this.llms = obj.llms;
    }

    static fromUnknown(value: unknown): DtoAssistantOptions {
        const obj = UtilsTypes.unknownToObject(value).unwrap();
        const llmsUnknown = UtilsTypes.unknownToArray(obj.llms).unwrap();
        return new DtoAssistantOptions({
            llms: llmsUnknown.map(DtoLlmData.fromUnknown)
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

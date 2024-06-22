import { UtilsTypes } from "@/utils/core/utils_types";

export class DocumentDto {
    page_content: String;
    metadata: Record<string, string>;

    constructor(obj: {
        page_content: String;
        metadata: Record<string, string>;
    }) {
        this.page_content = obj.page_content;
        this.metadata = obj.metadata;
    }

    static fromUnknown(value: unknown): DocumentDto {
        const obj = UtilsTypes.unknownToObject(value).unwrap();
        const metadataObj: any = UtilsTypes.unknownToObject(obj.metadata).unwrap();
        return new DocumentDto({
            page_content: UtilsTypes.unknownToString(obj.page_content).unwrap(),
            metadata: metadataObj
        })
    }

    static listFromUnknown(value: unknown): DocumentDto[] {
        const list = UtilsTypes.unknownToArray(value).unwrap();
        return list.map(DocumentDto.fromUnknown);
    }
}
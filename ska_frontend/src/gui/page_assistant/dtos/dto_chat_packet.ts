import { Err, Ok, type Result } from "@/utils/core/result";
import { UtilsTypes } from "@/utils/core/utils_types";
import { DocumentDto } from "./dto_document";

export class DtoChatPacket {
    created_at: number;
    message_body: string;
    packet_type: ChatPacketType;
    context: DocumentDto[];

    constructor(obj: {
        created_at: number,
        message_body: string,
        packet_type: ChatPacketType,
        context: DocumentDto[],
    }) {
        this.created_at = obj.created_at;
        this.message_body = obj.message_body;
        this.packet_type = obj.packet_type;
        this.context = obj.context;
    }

    static fromUnknown(value: unknown): DtoChatPacket {
        const obj = UtilsTypes.unknownToObject(value).unwrap();
        const context = DocumentDto.listFromUnknown(obj.context);
        const packet_type_str = UtilsTypes.unknownToString(obj.packet_type).unwrap();
        return new DtoChatPacket({
            created_at: UtilsTypes.unknownToNumber(obj.created_at).unwrap(),
            message_body: UtilsTypes.unknownToString(obj.message_body).unwrap(),
            packet_type: chatPacketTypeFromString(packet_type_str).unwrap(),
            context: context
        })
    }
}

export enum ChatPacketType {
    QUESTION = 'QUESTION',
    ANSWER = 'ANSWER',
}
export function chatPacketTypeFromString(value: string): Result<ChatPacketType, Error> {
    const enumValue = ChatPacketType[value as ChatPacketType] as ChatPacketType | undefined | null;
    if (enumValue == null) {
        return Err.new(new Error(`${value} does not exist in ChatPacketType`))
    } else {
        return Ok.new(enumValue);
    }
}
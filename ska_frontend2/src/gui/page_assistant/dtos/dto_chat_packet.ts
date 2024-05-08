import { Err, Ok, type Result } from "@/utils/core/result";
import { UtilsTypes } from "@/utils/core/utils_types";

export class DtoChatPacket {
    timestamp: number;
    value: string;
    packet_type: ChatPacketType;

    constructor(obj: {
        timestamp: number,
        value: string,
        packet_type: ChatPacketType,
    }) {
        this.timestamp = obj.timestamp;
        this.value = obj.value;
        this.packet_type = obj.packet_type;
    }

    static fromUnknown(value: unknown): DtoChatPacket {
        const obj = UtilsTypes.unknownToObject(value).unwrap();
        const packetTypeStr = UtilsTypes.unknownToString(obj.packet_type).unwrap();
        return new DtoChatPacket({
            timestamp: UtilsTypes.unknownToNumber(obj.timestamp).unwrap(),
            value: UtilsTypes.unknownToString(obj.value).unwrap(),
            packet_type: chatPacketTypeFromString(packetTypeStr).unwrap()
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
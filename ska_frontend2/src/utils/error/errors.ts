import type { Result } from "../core/result";
import { UtilsTypes } from "../core/utils_types";

export class DtoErrorResponse {
    status_code: number;
    is_unexpected_error: boolean;
    packets: DtoErrorPacket[];

    constructor(obj: {
        status_code: number;
        is_unexpected_error: boolean;
        packets: DtoErrorPacket[];
    }) {
        this.status_code = obj.status_code;
        this.is_unexpected_error = obj.is_unexpected_error;
        this.packets = obj.packets;

    }

    static fromUnknown(value: unknown): DtoErrorResponse {
        const obj = UtilsTypes.unknownToObject(value).unwrap();
        const packets = UtilsTypes.unknownToArray(obj.packets).unwrap()
        return new DtoErrorResponse({
            status_code: UtilsTypes.unknownToNumber(obj.status_code).unwrap(),
            is_unexpected_error: UtilsTypes.unknownToBoolean(obj.is_unexpected_error).unwrap(),
            packets: packets.map(DtoErrorPacket.fromUnknown)
        })
    }
}


export class DtoErrorPacket {
    message: string;

    constructor(obj: {
        message: string;
    }) {
        this.message = obj.message;
    }

    static fromUnknown(value: unknown): DtoErrorPacket {
        const obj = UtilsTypes.unknownToObject(value).unwrap();
        return new DtoErrorPacket({
            message: UtilsTypes.unknownToString(obj.message).unwrap(),
        })

    }
}

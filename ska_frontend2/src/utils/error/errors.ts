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
}


export class DtoErrorPacket {
    message: string;

    constructor(obj: {
        message: string;
    }) {
        this.message = obj.message;
    }
}

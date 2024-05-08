import { Err, Ok, type Result } from "./result";

export type UnknownObject = {
    [index: string | number | symbol]: unknown
}

export class UtilsTypes {
    static isString(value: unknown): value is string {
        return value != null && typeof value === 'string';
    }

    static isNumber(value: unknown): value is number {
        return value != null && typeof value === 'number';
    }

    static isBoolean(value: unknown): value is boolean {
        return value != null && typeof value === 'boolean';
    }

    static isArray(value: unknown): value is unknown[] {
        return value != null && typeof value === 'object' && Array.isArray(value);
    }

    static isUnknownObject(value: unknown): value is UnknownObject {
        return value != null && typeof value === 'object' && !Array.isArray(value);
    }


    static unknownToString(value: unknown): Result<string, Error> {
        if (this.isString(value)) {
            return Ok.new(value);
        } else {
            return Err.new(new Error(`Error unknownToString typeof value is ${typeof value}`))
        }
    }
    static unknownToStringNullable(value: unknown): Result<string | null, Error> {
        if (value == null) {
            return Ok.new(null);
        } else {
            return this.unknownToString(value);
        }
    }

    static unknownToNumber(value: unknown): Result<number, Error> {
        if (this.isNumber(value)) {
            return Ok.new(value);
        } else {
            return Err.new(new Error(`Error unknownToNumber typeof value is ${typeof value}`))
        }
    }
    static unknownToNumberNullable(value: unknown): Result<number | null, Error> {
        if (value == null) {
            return Ok.new(null);
        } else {
            return this.unknownToNumber(value);
        }
    }

    static unknownToBoolean(value: unknown): Result<boolean, Error> {
        if (this.isBoolean(value)) {
            return Ok.new(value);
        } else {
            return Err.new(new Error(`Error unknownToBoolean typeof value is ${typeof value}`))
        }
    }
    static unknownToBooleanNullable(value: unknown): Result<boolean | null, Error> {
        if (value == null) {
            return Ok.new(null);
        } else {
            return this.unknownToBoolean(value);
        }
    }

    static unknownToArray(value: unknown): Result<unknown[], Error> {
        if (this.isArray(value)) {
            return Ok.new(value);
        } else {
            return Err.new(new Error(`Error unknownToArray typeof value is ${typeof value}`))
        }
    }
    static unknownToArrayNullable(value: unknown): Result<unknown[] | null, Error> {
        if (value == null) {
            return Ok.new(null);
        } else {
            return this.unknownToArray(value);
        }
    }

    static unknownToObject(value: unknown): Result<UnknownObject, Error> {
        if (this.isUnknownObject(value)) {
            return Ok.new(value);
        } else {
            return Err.new(new Error(`Error unknownToObject typeof value is ${typeof value}`))
        }
    }
    static unknownToObjectNullable(value: unknown): Result<UnknownObject | null, Error> {
        if (value == null) {
            return Ok.new(null);
        } else {
            return this.unknownToObject(value);
        }
    }
}
import { Err, Ok, type Result } from "@/utils/core/result";
import { UtilsTypes } from "@/utils/core/utils_types";

export class DtoUserDetails {
    user_id: number;
    sub: string;
    username: string;
    email: string;
    roles: AuthRoles[];

    constructor(obj: {
        user_id: number,
        sub: string,
        username: string,
        email: string,
        roles: AuthRoles[],
    }) {
        this.user_id = obj.user_id;
        this.sub = obj.sub;
        this.username = obj.username;
        this.email = obj.email;
        this.roles = obj.roles;
    }


    static fromUnknown(value: unknown): DtoUserDetails {
        const obj = UtilsTypes.unknownToObject(value).unwrap();
        const roles = UtilsTypes.unknownToArray(obj.roles).unwrap()
        return new DtoUserDetails({
            user_id: UtilsTypes.unknownToNumber(obj.user_id).unwrap(),
            sub: UtilsTypes.unknownToString(obj.sub).unwrap(),
            username: UtilsTypes.unknownToString(obj.username).unwrap(),
            email: UtilsTypes.unknownToString(obj.email).unwrap(),
            roles: roles.map(roleUnknown => {
                const roleStr = UtilsTypes.unknownToString(roleUnknown).unwrap();
                return authRolesFromString(roleStr).unwrap();
            }),
        })
    }
}

export enum AuthRoles {
    SKA_ADMIN = 'SKA_ADMIN',
    SKA_USER = 'SKA_USER',
    SKA_GUEST = 'SKA_GUEST',
}

export function authRolesFromString(value: string): Result<AuthRoles, Error> {
    const enumValue = AuthRoles[value as AuthRoles] as AuthRoles | undefined | null;
    if (enumValue == null) {
        return Err.new(new Error(`${value} does not exist in BuildEnv`))
    } else {
        return Ok.new(enumValue);
    }
}

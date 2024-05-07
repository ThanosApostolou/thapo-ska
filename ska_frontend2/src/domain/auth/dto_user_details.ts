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
}

export enum AuthRoles {
    SKA_ADMIN = 'SKA_ADMIN',
    SKA_USER = 'SKA_USER',
    SKA_GUEST = 'SKA_GUEST',
}

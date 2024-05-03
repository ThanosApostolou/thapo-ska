import { Err, Ok, type Result } from "@/utils/core/result";

export enum BuildEnv {
    web = 'web',
    android = 'android',
    ios = 'ios',
    desktop = 'desktop'
}

export function buildEnvFromString(value: string): Result<BuildEnv, Error> {
    const enumValue = BuildEnv[value as BuildEnv] as BuildEnv | undefined | null;
    if (enumValue == null) {
        return Err.new(new Error(`${value} does not exist in BuildEnv`))
    } else {
        return Ok.new(enumValue);
    }
}
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub struct EnvConfig {
    pub env_profile: String,
}

impl EnvConfig {
    pub fn from_env() -> EnvConfig {
        let env_profile = String::from(env!("THAPO_SKA_ENV_PROFILE"));
        return EnvConfig { env_profile };
    }
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub struct EnvConfig {
    pub env_profile: String,
    pub base_href: String,
}

impl EnvConfig {
    pub fn from_env() -> EnvConfig {
        let env_profile = String::from(env!("THAPO_SKA_ENV_PROFILE"));
        let base_href = String::from(env!("THAPO_SKA_BASE_HREF"));
        return EnvConfig {
            env_profile,
            base_href,
        };
    }
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub struct EnvConfig {
    pub env_profile: String,
    pub base_href: String,
    pub backend_url: String,
}

impl EnvConfig {
    pub fn from_env() -> EnvConfig {
        let env_profile = String::from(env!("THAPO_SKA_ENV_PROFILE"));
        let base_href = String::from(env!("THAPO_SKA_BASE_HREF"));
        let backend_url = String::from(env!("THAPO_SKA_BACKEND_URL"));
        EnvConfig {
            env_profile,
            base_href,
            backend_url,
        }
    }
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub struct EnvConfig {
    pub env_profile: String,
    pub base_href: String,
    pub frontend_url: String,
    pub backend_url: String,
    pub auth_issuer_url: String,
    pub auth_client_id: String,
    pub auth_client_secret: String,
}

impl EnvConfig {
    pub fn from_env() -> EnvConfig {
        let env_profile = String::from(env!("THAPO_SKA_ENV_PROFILE"));
        let base_href = String::from(env!("THAPO_SKA_BASE_HREF"));
        let frontend_url = String::from(env!("THAPO_SKA_FRONTEND_URL"));
        let backend_url = String::from(env!("THAPO_SKA_BACKEND_URL"));
        let auth_issuer_url = String::from(env!("THAPO_SKA_AUTH_ISSUER_URL"));
        let auth_client_id = String::from(env!("THAPO_SKA_AUTH_CLIENT_ID"));
        let auth_client_secret = String::from(env!("THAPO_SKA_AUTH_CLIENT_SECRET"));
        EnvConfig {
            env_profile,
            base_href,
            frontend_url,
            backend_url,
            auth_issuer_url,
            auth_client_id,
            auth_client_secret,
        }
    }
}

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub struct EnvConfig {
    pub env_profile: String,
    pub server_host: String,
    pub server_port: u16,
}

impl EnvConfig {
    pub fn from_env() -> EnvConfig {
        let env_profile = dotenv::var("THAPO_SKA_ENV_PROFILE")
            .expect("THAPO_SKA_ENV_PROFILE env var is missing");
        let server_host = dotenv::var("SERVER_HOST").expect("SERVER_HOST env var is missing");
        let server_port_str = dotenv::var("SERVER_PORT").expect("SERVER_PORT env var is missing");
        let server_port = server_port_str
            .parse::<u16>()
            .expect("SERVER_PORT was not a usize");
        EnvConfig {
            env_profile,
            server_host,
            server_port,
        }
    }
}

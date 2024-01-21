#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub struct EnvConfig {
    pub env_profile: String,
    pub server_host: String,
    pub server_port: u16,
    pub db_host: String,
    pub db_port: u16,
    pub db_database: String,
    pub db_schema: String,
}

impl EnvConfig {
    pub fn from_env() -> EnvConfig {
        let env_profile =
            dotenv::var("THAPO_SKA_ENV_PROFILE").expect("THAPO_SKA_ENV_PROFILE env var is missing");
        let server_host =
            dotenv::var("THAPO_SKA_SERVER_HOST").expect("THAPO_SKA_SERVER_HOST env var is missing");
        let server_port_str =
            dotenv::var("THAPO_SKA_SERVER_PORT").expect("THAPO_SKA_SERVER_PORT env var is missing");
        let server_port = server_port_str
            .parse::<u16>()
            .expect("SERVER_PORT was not a usize");
        let db_host =
            dotenv::var("THAPO_SKA_DB_HOST").expect("THAPO_SKA_DB_HOST env var is missing");
        let db_port_str =
            dotenv::var("THAPO_SKA_DB_PORT").expect("THAPO_SKA_DB_PORT env var is missing");
        let db_port = db_port_str
            .parse::<u16>()
            .expect("THAPO_SKA_DB_PORT was not a usize");
        let db_database =
            dotenv::var("THAPO_SKA_DB_DATABASE").expect("THAPO_SKA_DB_DATABASE env var is missing");
        let db_schema =
            dotenv::var("THAPO_SKA_DB_SCHEMA").expect("THAPO_SKA_DB_SCHEMA env var is missing");

        EnvConfig {
            env_profile,
            server_host,
            server_port,
            db_host,
            db_port,
            db_database,
            db_schema,
        }
    }
}

use std::env;

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub struct EnvConfig {
    pub rust_log: String,
    pub env_profile: String,
    pub server_path: String,
    pub server_host: String,
    pub server_port: u16,
    pub db_host: String,
    pub db_port: u16,
    pub db_database: String,
    pub db_schema: String,
    pub auth_issuer_url: String,
    pub auth_client_id: String,
    pub request_timeout: u64,
    pub ska_conf_dir: String,
    pub ska_data_dir: String,
    pub ska_tmp_dir: String,
    pub ska_user_conf_dir: String,
    pub ska_llm_dir: String,
}

impl EnvConfig {
    pub fn from_env() -> EnvConfig {
        let ska_conf_dir =
            dotenv::var("THAPO_SKA_CONF_DIR").expect("THAPO_SKA_CONF_DIR env var is missing");

        let dotenv_file = ska_conf_dir.clone() + "/.env";
        dotenv::from_filename(&dotenv_file)
            .unwrap_or_else(|_| panic!("could not load file {}", dotenv_file.clone()));

        let ska_data_dir =
            dotenv::var("THAPO_SKA_DATA_DIR").expect("THAPO_SKA_DATA_DIR env var is missing");
        let ska_tmp_dir =
            dotenv::var("THAPO_SKA_TMP_DIR").expect("THAPO_SKA_TMP_DIR env var is missing");
        let ska_user_conf_dir = dotenv::var("THAPO_SKA_USER_CONF_DIR")
            .expect("THAPO_SKA_USER_CONF_DIR env var is missing");
        let ska_llm_dir =
            dotenv::var("THAPO_SKA_LLM_DIR").expect("THAPO_SKA_LLM_DIR env var is missing");
        let rust_log: String = dotenv::var("RUST_LOG").expect("RUST_LOG env var is missing");
        let env_profile =
            dotenv::var("THAPO_SKA_ENV_PROFILE").expect("THAPO_SKA_ENV_PROFILE env var is missing");
        let server_path =
            dotenv::var("THAPO_SKA_SERVER_PATH").expect("THAPO_SKA_SERVER_PATH env var is missing");
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
        let auth_issuer_url = dotenv::var("THAPO_SKA_AUTH_ISSUER_URL")
            .expect("THAPO_SKA_AUTH_ISSUER_URL env var is missing");
        let auth_client_id = dotenv::var("THAPO_SKA_AUTH_CLIENT_ID")
            .expect("THAPO_SKA_AUTH_CLIENT_ID env var is missing");
        let request_timeout_str = dotenv::var("THAPO_SKA_REQUEST_TIMEOUT")
            .expect("THAPO_SKA_REQUEST_TIMEOUT env var is missing");
        let request_timeout = request_timeout_str
            .parse::<u64>()
            .expect("THAPO_SKA_REQUEST_TIMEOUT was not a usize");

        EnvConfig {
            rust_log,
            env_profile,
            server_path,
            server_host,
            server_port,
            db_host,
            db_port,
            db_database,
            db_schema,
            auth_issuer_url,
            auth_client_id,
            request_timeout,
            ska_conf_dir,
            ska_data_dir,
            ska_tmp_dir,
            ska_user_conf_dir,
            ska_llm_dir,
        }
    }
}

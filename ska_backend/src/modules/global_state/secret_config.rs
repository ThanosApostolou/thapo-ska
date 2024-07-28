#[derive(Clone)]
pub struct SecretConfig {
    pub db_user: String,
    pub db_password: String,
    pub auth_client_secret: String,
}

impl SecretConfig {
    pub fn from_env() -> SecretConfig {
        let db_user =
            dotenvy::var("THAPO_SKA_DB_USER").expect("THAPO_SKA_DB_USER env var is missing");
        let db_password = dotenvy::var("THAPO_SKA_DB_PASSWORD")
            .expect("THAPO_SKA_DB_PASSWORD env var is missing");
        let auth_client_secret = dotenvy::var("THAPO_SKA_AUTH_CLIENT_SECRET")
            .expect("THAPO_SKA_AUTH_CLIENT_SECRET env var is missing");

        SecretConfig {
            db_user,
            db_password,
            auth_client_secret,
        }
    }
}

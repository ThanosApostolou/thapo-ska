#[derive(Clone)]
pub struct SecretConfig {
    pub db_user: String,
    pub db_password: String,
}

impl SecretConfig {
    pub fn from_env() -> SecretConfig {
        let db_user =
            dotenv::var("THAPO_SKA_DB_USER").expect("THAPO_SKA_DB_USER env var is missing");
        let db_password =
            dotenv::var("THAPO_SKA_DB_PASSWORD").expect("THAPO_SKA_DB_PASSWORD env var is missing");

        SecretConfig {
            db_user,
            db_password,
        }
    }
}

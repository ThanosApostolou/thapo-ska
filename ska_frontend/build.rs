use dotenv::from_filename;
use std::env;

fn main() {
    let env_file = env::var("THAPO_SKA_ENV_FILE").unwrap_or_else(|_| ".env.local".to_string());
    from_filename(env_file.clone()).ok();
    let dotenv_path = dotenv::from_filename(env_file).expect("failed to find .env file");
    println!("cargo:rerun-if-changed={}", dotenv_path.display());

    for env_var in dotenv::vars() {
        let (key, value) = env_var;
        println!("cargo:rustc-env={key}={value}");
    }
}

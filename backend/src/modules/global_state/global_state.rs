use super::EnvConfig;

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug, Hash)]
pub struct GlobalState {
    pub env_config: EnvConfig,
}

impl GlobalState {
    pub fn initialize_default() -> GlobalState {
        let env_config = EnvConfig::from_env();
        GlobalState { env_config }
    }
}

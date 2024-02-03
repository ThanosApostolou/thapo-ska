use leptos::{create_rw_signal, RwSignal};

use crate::modules::auth::UserDetails;

#[derive(Clone, Debug)]
pub struct GlobalStore {
    pub user_details: RwSignal<Option<UserDetails>>,
}

impl GlobalStore {
    pub fn initialize_default() -> GlobalStore {
        let user_details: RwSignal<Option<UserDetails>> = create_rw_signal(Option::None);
        GlobalStore { user_details }
    }
}

use leptos::{Signal, WriteSignal};
use leptos_use::{
    storage::{use_local_storage, use_session_storage},
    utils::FromToStringCodec,
};

pub fn use_session_pkce_verifier() -> (Signal<String>, WriteSignal<String>, impl Fn() + Clone) {
    let (session_pkce_verifier, session_set_pkce_verifier, session_remove_pkce_verifier) =
        use_session_storage::<String, FromToStringCodec>("pkce_verifier");
    (
        session_pkce_verifier,
        session_set_pkce_verifier,
        session_remove_pkce_verifier,
    )
}

pub fn use_storage_refresh_token() -> (Signal<String>, WriteSignal<String>, impl Fn() + Clone) {
    let (storage_refresh_token, storage_set_refresh_token, storage_remove_refresh_token) =
        use_local_storage::<String, FromToStringCodec>("refresh_token");
    (
        storage_refresh_token,
        storage_set_refresh_token,
        storage_remove_refresh_token,
    )
}

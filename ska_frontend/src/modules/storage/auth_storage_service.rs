use leptos::{Signal, WriteSignal};
use leptos_use::{storage::use_session_storage, utils::FromToStringCodec};

pub fn use_session_pkce_verifier() -> (Signal<String>, WriteSignal<String>, impl Fn() + Clone) {
    let (session_pkce_verifier, session_set_pkce_verifier, session_remove_pkce_verifier) =
        use_session_storage::<String, FromToStringCodec>("pkce_verifier");
    (
        session_pkce_verifier,
        session_set_pkce_verifier,
        session_remove_pkce_verifier,
    )
}

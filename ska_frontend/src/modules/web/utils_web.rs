use leptos::*;
use reqwest::RequestBuilder;

use crate::modules::global_state::GlobalStore;

pub fn add_common_headers(
    global_store: &GlobalStore,
    request_builder: RequestBuilder,
) -> RequestBuilder {
    let mut request_builder = request_builder;
    if let Some(access_token) = global_store.access_token.get_untracked() {
        let access_token_str = "Bearer ".to_string() + access_token.secret().as_ref();
        request_builder = request_builder.header("Authorization", access_token_str)
    }
    request_builder
}

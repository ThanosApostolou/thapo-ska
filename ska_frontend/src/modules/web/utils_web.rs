use leptos::*;
use reqwest::{RequestBuilder, StatusCode};
use serde::Deserialize;

use crate::modules::{
    error::{DtoErrorPacket, DtoErrorResponse},
    global_state::GlobalStore,
};

pub fn request_builder_with_headers(
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

pub async fn send_request<DTO>(
    global_store: &GlobalStore,
    request_builder: RequestBuilder,
) -> Result<DTO, DtoErrorResponse>
where
    DTO: for<'a> Deserialize<'a>,
{
    let request_builder = request_builder_with_headers(global_store, request_builder);

    let http_response = request_builder.send().await.map_err(|error| {
        let error_status = error.status().unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
        DtoErrorResponse::new(error_status.as_u16(), true, vec![])
    })?;

    if http_response.status().is_success() {
        let response = http_response.json::<DTO>().await.map_err(|error| {
            DtoErrorResponse::new(
                error
                    .status()
                    .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
                    .as_u16(),
                true,
                vec![DtoErrorPacket::new(
                    "could not deserialize response".to_string(),
                )],
            )
        })?;
        return Ok(response);
    }
    match http_response.status() {
        StatusCode::UNPROCESSABLE_ENTITY => {
            let response = http_response
                .json::<DtoErrorResponse>()
                .await
                .map_err(|error| {
                    DtoErrorResponse::new(
                        error
                            .status()
                            .unwrap_or(StatusCode::INTERNAL_SERVER_ERROR)
                            .as_u16(),
                        true,
                        vec![DtoErrorPacket::new(
                            "could not deserialize DtoErrorResponse".to_string(),
                        )],
                    )
                })?;
            Err(response)
        }
        _ => Err(DtoErrorResponse::new(
            http_response.status().as_u16(),
            true,
            vec![DtoErrorPacket::new("unexpected error".to_string())],
        )),
    }
}

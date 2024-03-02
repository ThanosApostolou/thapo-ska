use axum::http::HeaderValue;
use hyper::HeaderMap;
use oauth2::{
    basic::{BasicErrorResponseType, BasicTokenType},
    AccessToken, ClientId, ClientSecret, EmptyExtraTokenFields, ExtraTokenFields, IntrospectionUrl,
    RevocationErrorResponseType, StandardErrorResponse, StandardRevocableToken,
    StandardTokenIntrospectionResponse, StandardTokenResponse, TokenIntrospectionResponse,
};
use openidconnect::{
    core::{
        CoreAuthDisplay, CoreAuthPrompt, CoreGenderClaim, CoreJsonWebKey, CoreJsonWebKeyType,
        CoreJsonWebKeyUse, CoreJweContentEncryptionAlgorithm, CoreJwsSigningAlgorithm,
    },
    Client, EmptyAdditionalClaims, IdTokenFields, IssuerUrl, ProviderMetadata,
    ProviderMetadataWithLogout,
};
use serde::{Deserialize, Serialize};

use crate::modules::global_state::{EnvConfig, GlobalState, SecretConfig};
use std::sync::Arc;

pub type MyStandardTokenResponse = StandardTokenResponse<
    IdTokenFields<
        EmptyAdditionalClaims,
        EmptyExtraTokenFields,
        CoreGenderClaim,
        CoreJweContentEncryptionAlgorithm,
        CoreJwsSigningAlgorithm,
        CoreJsonWebKeyType,
    >,
    BasicTokenType,
>;

pub type MyProviderMetadata = ProviderMetadata<
    openidconnect::LogoutProviderMetadata<openidconnect::EmptyAdditionalProviderMetadata>,
    CoreAuthDisplay,
    openidconnect::core::CoreClientAuthMethod,
    openidconnect::core::CoreClaimName,
    openidconnect::core::CoreClaimType,
    openidconnect::core::CoreGrantType,
    CoreJweContentEncryptionAlgorithm,
    openidconnect::core::CoreJweKeyManagementAlgorithm,
    CoreJwsSigningAlgorithm,
    CoreJsonWebKeyType,
    CoreJsonWebKeyUse,
    CoreJsonWebKey,
    openidconnect::core::CoreResponseMode,
    openidconnect::core::CoreResponseType,
    openidconnect::core::CoreSubjectIdentifierType,
>;

pub type MyOidcClient = Client<
    EmptyAdditionalClaims,
    CoreAuthDisplay,
    CoreGenderClaim,
    CoreJweContentEncryptionAlgorithm,
    CoreJwsSigningAlgorithm,
    CoreJsonWebKeyType,
    CoreJsonWebKeyUse,
    CoreJsonWebKey,
    CoreAuthPrompt,
    StandardErrorResponse<BasicErrorResponseType>,
    MyStandardTokenResponse,
    BasicTokenType,
    StandardTokenIntrospectionResponse<MyExtraTokenFields, BasicTokenType>,
    StandardRevocableToken,
    StandardErrorResponse<RevocationErrorResponseType>,
>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyExtraTokenFields {
    pub realm_access: RealmAccess,
}

impl ExtraTokenFields for MyExtraTokenFields {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealmAccess {
    pub roles: Vec<String>,
}

// const PATH_API_AUTH: &'static str = "/auth";

pub async fn create_oidc_client(
    env_config: &EnvConfig,
    secret_config: &SecretConfig,
) -> anyhow::Result<(MyProviderMetadata, MyOidcClient)> {
    let provider_metadata = ProviderMetadataWithLogout::discover_async(
        IssuerUrl::new(env_config.auth_issuer_url.clone())?,
        my_async_http_client,
    )
    .await?;

    // let log = LogoutProviderMetadata

    // Create an OpenID Connect client by specifying the client ID, client secret, authorization URL
    // and token URL.
    // Some(ClientSecret::new("client_secret".to_string()))

    let introspection_url = IntrospectionUrl::new(env_config.auth_introspection_url.clone())?;
    let client = Client::from_provider_metadata(
        provider_metadata.clone(),
        ClientId::new(env_config.auth_client_id.clone()),
        Some(ClientSecret::new(secret_config.auth_client_secret.clone())),
    )
    .set_introspection_uri(introspection_url);

    Ok((provider_metadata, client))
}

pub async fn introspect(
    oidc_client: &MyOidcClient,
    access_token_str: String,
) -> anyhow::Result<StandardTokenIntrospectionResponse<MyExtraTokenFields, BasicTokenType>> {
    let access_token: AccessToken = AccessToken::new(access_token_str);
    let response: StandardTokenIntrospectionResponse<MyExtraTokenFields, BasicTokenType> =
        oidc_client
            .introspect(&access_token)?
            .request_async(my_async_http_client)
            .await?;
    tracing::info!("roles: {:?}", response.extra_fields().realm_access.roles);
    tracing::info!("active: {:?}", response.active());
    Ok(response)
}

pub async fn authenticate_user(
    global_state: Arc<GlobalState>,
    headers: &HeaderMap,
) -> anyhow::Result<StandardTokenIntrospectionResponse<MyExtraTokenFields, BasicTokenType>> {
    let empty = HeaderValue::from_str("empty").unwrap();
    let header_authorization = headers.get("Authorization").unwrap_or(&empty).to_str()?;
    // tracing::info!("headers {}", header_authorization);
    let access_token_str = header_authorization
        .strip_prefix("Bearer ")
        .ok_or(anyhow::anyhow!("no bearer"))?;
    let response = introspect(&global_state.oidc_client, access_token_str.to_string()).await?;
    let ef: &MyExtraTokenFields = response.extra_fields();
    Ok(response)
}

pub async fn my_async_http_client(
    request: oauth2::HttpRequest,
) -> Result<oauth2::HttpResponse, oauth2::reqwest::Error<reqwest::Error>> {
    let client = {
        let builder = reqwest::Client::builder().danger_accept_invalid_certs(true);

        // Following redirects opens the client up to SSRF vulnerabilities.
        // but this is not possible to prevent on wasm targets
        #[cfg(not(target_arch = "wasm32"))]
        let builder = builder.redirect(reqwest::redirect::Policy::none());

        builder.build().map_err(oauth2::reqwest::Error::Reqwest)?
    };

    let mut request_builder = client
        .request(request.method, request.url.as_str())
        .body(request.body);
    for (name, value) in &request.headers {
        request_builder = request_builder.header(name.as_str(), value.as_bytes());
    }
    let request = request_builder
        .build()
        .map_err(oauth2::reqwest::Error::Reqwest)?;

    let response = client
        .execute(request)
        .await
        .map_err(oauth2::reqwest::Error::Reqwest)?;

    let status_code = response.status();
    let headers = response.headers().to_owned();
    let chunks = response
        .bytes()
        .await
        .map_err(oauth2::reqwest::Error::Reqwest)?;
    Ok(oauth2::HttpResponse {
        status_code,
        headers,
        body: chunks.to_vec(),
    })
}

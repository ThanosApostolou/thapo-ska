use anyhow::anyhow;
use leptos::*;
use oauth2::{
    basic::{BasicErrorResponseType, BasicTokenType},
    ClientSecret, EmptyExtraTokenFields, PkceCodeVerifier, RefreshToken,
    RevocationErrorResponseType, StandardErrorResponse, StandardRevocableToken,
    StandardTokenIntrospectionResponse, StandardTokenResponse,
};
use openidconnect::{
    core::{
        CoreAuthDisplay, CoreAuthPrompt, CoreAuthenticationFlow, CoreGenderClaim, CoreJsonWebKey,
        CoreJsonWebKeyType, CoreJsonWebKeyUse, CoreJweContentEncryptionAlgorithm,
        CoreJwsSigningAlgorithm,
    },
    reqwest::async_http_client,
    AuthorizationCode, Client, ClientId, CsrfToken, EmptyAdditionalClaims, IdTokenFields,
    IssuerUrl, Nonce, PkceCodeChallenge, ProviderMetadata, ProviderMetadataWithLogout, RedirectUrl,
    TokenResponse,
};
// Use OpenID Connect Discovery to fetch the provider metadata.
use openidconnect::OAuth2TokenResponse;
use reqwest::Url;

use crate::modules::global_state::{EnvConfig, GlobalStore};

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
    StandardTokenIntrospectionResponse<EmptyExtraTokenFields, BasicTokenType>,
    StandardRevocableToken,
    StandardErrorResponse<RevocationErrorResponseType>,
>;

pub async fn create_oidc_client(
    env_config: &EnvConfig,
) -> anyhow::Result<(MyProviderMetadata, MyOidcClient)> {
    let provider_metadata = ProviderMetadataWithLogout::discover_async(
        IssuerUrl::new(env_config.auth_issuer_url.clone())?,
        async_http_client,
    )
    .await?;

    // let log = LogoutProviderMetadata

    // Create an OpenID Connect client by specifying the client ID, client secret, authorization URL
    // and token URL.
    // Some(ClientSecret::new("client_secret".to_string()))
    let client = Client::from_provider_metadata(
        provider_metadata.clone(),
        ClientId::new(env_config.auth_client_id.clone()),
        Some(ClientSecret::new(env_config.auth_client_secret.clone())),
    )
    // Set the URL the user will be redirected to after the authorization process.
    .set_redirect_uri(RedirectUrl::new(
        env_config.frontend_url.clone() + "/login",
    )?);
    Ok((provider_metadata, client))
}

pub async fn get_auth_url(client: &MyOidcClient) -> (Url, CsrfToken, Nonce, PkceCodeVerifier) {
    // Generate a PKCE challenge.
    let (pkce_challenge, pkce_verifier) = PkceCodeChallenge::new_random_sha256();

    // Generate the full authorization URL.
    let (auth_url, csrf_token, nonce) = client
        .authorize_url(
            CoreAuthenticationFlow::AuthorizationCode,
            CsrfToken::new_random,
            Nonce::new_random,
        )
        // Set the desired scopes.
        // .add_scope(Scope::new("read".to_string()))
        // .add_scope(Scope::new("write".to_string()))
        // Set the PKCE code challenge.
        .set_pkce_challenge(pkce_challenge)
        .url();

    // This is the URL you should redirect the user to, in order to trigger the authorization
    // process.
    println!("Browse to: {}", auth_url);
    (auth_url, csrf_token, nonce, pkce_verifier)
}

pub async fn exchange_code(
    client: &MyOidcClient,
    pkce_verifier: PkceCodeVerifier,
    code: String,
) -> anyhow::Result<MyStandardTokenResponse> {
    // Now you can exchange it for an access token and ID token.
    let token_response = client
        .exchange_code(AuthorizationCode::new(code))
        // Set the PKCE code verifier.
        .set_pkce_verifier(pkce_verifier)
        .request_async(async_http_client)
        .await?;
    Ok(token_response)
}

/// exchanges a refresh_token for a MyStandardTokenResponse
pub async fn exchange_refresh_token(
    oidc_client: &MyOidcClient,
    refresh_token: &RefreshToken,
) -> anyhow::Result<MyStandardTokenResponse> {
    let response: MyStandardTokenResponse = oidc_client
        .exchange_refresh_token(refresh_token)
        .request_async(async_http_client)
        .await?;
    let refresh_token = match response.refresh_token() {
        Some(rt) => rt.secret().clone(),
        None => "".to_string(),
    };
    let id_token = match response.id_token() {
        Some(it) => it.to_string(),
        None => "".to_string(),
    };
    let access_token = response.access_token().secret().clone();
    log::info!("exchange refresh_token={}", refresh_token);
    log::info!("exchange id_token={}", id_token);
    log::info!("exchange access_token={}", access_token);
    Ok(response)
}

pub fn store_token_response(
    global_store: RwSignal<GlobalStore>,
    token_response: &MyStandardTokenResponse,
    storage_set_refresh_token: WriteSignal<String>,
) -> anyhow::Result<()> {
    let refresh_token = token_response
        .refresh_token()
        .ok_or(anyhow!("refresh token missing"))?;
    let id_token = token_response
        .id_token()
        .ok_or(anyhow!("id token missing"))?;
    let access_token = token_response.access_token();
    // let access_token = token_response.access_token().clone();
    global_store
        .get()
        .refresh_token
        .set(Some(refresh_token.clone()));
    global_store.get().id_token.set(Some(id_token.clone()));
    global_store
        .get()
        .access_token
        .set(Some(access_token.clone()));
    storage_set_refresh_token(refresh_token.secret().clone());
    Ok(())
}

pub async fn initial_check_login(
    global_store: RwSignal<GlobalStore>,
    storage_refresh_token: Signal<String>,
    storage_set_refresh_token: WriteSignal<String>,
    oidc_client: &MyOidcClient,
) -> anyhow::Result<()> {
    let refresh_token_str = storage_refresh_token();
    if !refresh_token_str.is_empty() {
        let refresh_token = RefreshToken::new(refresh_token_str);
        let token_response = exchange_refresh_token(oidc_client, &refresh_token).await;
        match token_response {
            Ok(token_response) => {
                store_token_response(global_store, &token_response, storage_set_refresh_token)?;
            }
            Err(err) => {
                storage_set_refresh_token("".to_string());
                return Err(err);
            }
        };
    }
    Ok(())
}

pub async fn login(oidc_client: &MyOidcClient, session_set_pkce_verifier: WriteSignal<String>) {
    let (url, _, _, pkce_verifier) = get_auth_url(oidc_client).await;
    session_set_pkce_verifier(pkce_verifier.secret().clone());
    log::info!("url={}", url.clone());
    window().location().set_href(url.as_str()).unwrap();
}

pub async fn after_login(
    global_store: RwSignal<GlobalStore>,
    oidc_client: &MyOidcClient,
    session_pkce_verifier: Signal<String>,
    session_set_pkce_verifier: WriteSignal<String>,
    storage_set_refresh_token: WriteSignal<String>,
    iss: Option<String>,
    state: Option<String>,
    code: Option<String>,
) -> anyhow::Result<()> {
    if global_store.get().refresh_token.get().is_none() {
        if let Some(_) = iss {
            if let Some(_) = state {
                if let Some(code) = code {
                    let pkce_verifier = session_pkce_verifier.get();
                    log::info!("pkce_verifier_String={}", pkce_verifier);

                    let token_response =
                        exchange_code(oidc_client, PkceCodeVerifier::new(pkce_verifier), code)
                            .await?;
                    store_token_response(global_store, &token_response, storage_set_refresh_token)?;

                    session_set_pkce_verifier("".to_string());
                }
            }
        }
    }
    Ok(())
}

pub async fn logout(
    env_config: &EnvConfig,
    oidc_provider_metadata: &MyProviderMetadata,
    storage_set_refresh_token: WriteSignal<String>,
) -> anyhow::Result<()> {
    let end_session_endpoint = oidc_provider_metadata
        .additional_metadata()
        .end_session_endpoint
        .clone()
        .ok_or(anyhow!("None end_session_endpoint"))?;
    log::info!("end_session_endpoint={:?}", end_session_endpoint);
    let mut final_url = Url::parse(end_session_endpoint.as_str())?;
    final_url
        .query_pairs_mut()
        .append_pair("client_id", env_config.auth_client_id.as_str())
        .append_pair(
            "post_logout_redirect_uri",
            (env_config.frontend_url.clone() + "/home").as_str(),
        );
    storage_set_refresh_token("".to_string());
    window()
        .location()
        .set_href(final_url.as_str())
        .map_err(|err| {
            anyhow!(
                "window set_href error: {}",
                err.as_string().unwrap_or("None".to_string())
            )
        })?;
    Ok(())
}

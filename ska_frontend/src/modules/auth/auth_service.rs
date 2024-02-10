use anyhow::{anyhow, Result};
use leptos::*;
use oauth2::{
    basic::{BasicErrorResponseType, BasicTokenType},
    EmptyExtraTokenFields, PkceCodeVerifier, RefreshToken, RevocationErrorResponseType,
    StandardErrorResponse, StandardRevocableToken, StandardTokenIntrospectionResponse,
    StandardTokenResponse,
};
use openidconnect::{
    core::{
        CoreAuthDisplay, CoreAuthPrompt, CoreAuthenticationFlow, CoreClaimName, CoreClaimType,
        CoreClientAuthMethod, CoreGenderClaim, CoreGrantType, CoreJsonWebKey, CoreJsonWebKeyType,
        CoreJsonWebKeyUse, CoreJweContentEncryptionAlgorithm, CoreJweKeyManagementAlgorithm,
        CoreJwsSigningAlgorithm, CoreResponseMode, CoreResponseType, CoreSubjectIdentifierType,
    },
    reqwest::async_http_client,
    AuthorizationCode, Client, ClientId, CsrfToken, EmptyAdditionalClaims,
    EmptyAdditionalProviderMetadata, IdTokenFields, IssuerUrl, LogoutProviderMetadata, Nonce,
    PkceCodeChallenge, ProviderMetadata, ProviderMetadataWithLogout, RedirectUrl,
};

// Use OpenID Connect Discovery to fetch the provider metadata.
use openidconnect::OAuth2TokenResponse;
use reqwest::Url;

use crate::modules::global_state::GlobalState;

pub async fn create_oidc_client() -> anyhow::Result<(
    ProviderMetadata<
        LogoutProviderMetadata<EmptyAdditionalProviderMetadata>,
        CoreAuthDisplay,
        CoreClientAuthMethod,
        CoreClaimName,
        CoreClaimType,
        CoreGrantType,
        CoreJweContentEncryptionAlgorithm,
        CoreJweKeyManagementAlgorithm,
        CoreJwsSigningAlgorithm,
        CoreJsonWebKeyType,
        CoreJsonWebKeyUse,
        CoreJsonWebKey,
        CoreResponseMode,
        CoreResponseType,
        CoreSubjectIdentifierType,
    >,
    Client<
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
        StandardTokenResponse<
            IdTokenFields<
                EmptyAdditionalClaims,
                EmptyExtraTokenFields,
                CoreGenderClaim,
                CoreJweContentEncryptionAlgorithm,
                CoreJwsSigningAlgorithm,
                CoreJsonWebKeyType,
            >,
            BasicTokenType,
        >,
        BasicTokenType,
        StandardTokenIntrospectionResponse<EmptyExtraTokenFields, BasicTokenType>,
        StandardRevocableToken,
        StandardErrorResponse<RevocationErrorResponseType>,
    >,
)> {
    let provider_metadata = ProviderMetadataWithLogout::discover_async(
        IssuerUrl::new(
            "https://thapo-ska-local.thapo-local.org:9443/iam/realms/thapo_ska_local".to_string(),
        )?,
        async_http_client,
    )
    .await?;

    // let log = LogoutProviderMetadata

    // Create an OpenID Connect client by specifying the client ID, client secret, authorization URL
    // and token URL.
    // Some(ClientSecret::new("client_secret".to_string()))
    let client = Client::from_provider_metadata(
        provider_metadata.clone(),
        ClientId::new("thapo_ska_local_frontend".to_string()),
        None,
    )
    // Set the URL the user will be redirected to after the authorization process.
    .set_redirect_uri(RedirectUrl::new(
        "https://thapo-ska-local.thapo-local.org:9443/app/home".to_string(),
    )?);
    Ok((provider_metadata, client))
}

pub async fn get_auth_url(
    client: &Client<
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
        StandardTokenResponse<
            IdTokenFields<
                EmptyAdditionalClaims,
                EmptyExtraTokenFields,
                CoreGenderClaim,
                CoreJweContentEncryptionAlgorithm,
                CoreJwsSigningAlgorithm,
                CoreJsonWebKeyType,
            >,
            BasicTokenType,
        >,
        BasicTokenType,
        StandardTokenIntrospectionResponse<EmptyExtraTokenFields, BasicTokenType>,
        StandardRevocableToken,
        StandardErrorResponse<RevocationErrorResponseType>,
    >,
) -> (Url, CsrfToken, Nonce, PkceCodeVerifier) {
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

pub async fn get_token_response(
    client: &Client<
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
        StandardTokenResponse<
            IdTokenFields<
                EmptyAdditionalClaims,
                EmptyExtraTokenFields,
                CoreGenderClaim,
                CoreJweContentEncryptionAlgorithm,
                CoreJwsSigningAlgorithm,
                CoreJsonWebKeyType,
            >,
            BasicTokenType,
        >,
        BasicTokenType,
        StandardTokenIntrospectionResponse<EmptyExtraTokenFields, BasicTokenType>,
        StandardRevocableToken,
        StandardErrorResponse<RevocationErrorResponseType>,
    >,
    pkce_verifier: PkceCodeVerifier,
    code: String,
) -> Result<
    StandardTokenResponse<
        IdTokenFields<
            EmptyAdditionalClaims,
            EmptyExtraTokenFields,
            CoreGenderClaim,
            CoreJweContentEncryptionAlgorithm,
            CoreJwsSigningAlgorithm,
            CoreJsonWebKeyType,
        >,
        BasicTokenType,
    >,
> {
    // Now you can exchange it for an access token and ID token.
    let token_response = client
        .exchange_code(AuthorizationCode::new(code))
        // Set the PKCE code verifier.
        .set_pkce_verifier(pkce_verifier)
        .request_async(async_http_client)
        .await?;
    Ok(token_response)
}

pub async fn login(
    global_state: ReadSignal<GlobalState>,
    session_set_pkce_verifier: WriteSignal<String>,
) {
    let (url, csrf_token, nonce, pkce_verifier) =
        get_auth_url(&global_state.get().oidc_client).await;
    session_set_pkce_verifier(pkce_verifier.secret().clone());
    log::info!("url={}", url.clone());
    window().location().set_href(url.as_str()).unwrap();
}

pub async fn check_if_user_logged(
    global_state: ReadSignal<GlobalState>,
    session_set_pkce_verifier: WriteSignal<String>,
    refresh_token: &RefreshToken,
) -> anyhow::Result<
    StandardTokenResponse<
        IdTokenFields<
            EmptyAdditionalClaims,
            EmptyExtraTokenFields,
            CoreGenderClaim,
            CoreJweContentEncryptionAlgorithm,
            CoreJwsSigningAlgorithm,
            CoreJsonWebKeyType,
        >,
        BasicTokenType,
    >,
> {
    let response: StandardTokenResponse<
        IdTokenFields<
            EmptyAdditionalClaims,
            EmptyExtraTokenFields,
            CoreGenderClaim,
            CoreJweContentEncryptionAlgorithm,
            CoreJwsSigningAlgorithm,
            CoreJsonWebKeyType,
        >,
        BasicTokenType,
    > = global_state
        .get()
        .oidc_client
        .exchange_refresh_token(refresh_token)
        .request_async(async_http_client)
        .await?;
    let x = response.refresh_token();
    Ok(response)
}

pub async fn logout(
    global_state: ReadSignal<GlobalState>,
) -> anyhow::Result<
    // StandardTokenResponse<
    //     IdTokenFields<
    //         EmptyAdditionalClaims,
    //         EmptyExtraTokenFields,
    //         CoreGenderClaim,
    //         CoreJweContentEncryptionAlgorithm,
    //         CoreJwsSigningAlgorithm,
    //         CoreJsonWebKeyType,
    //     >,
    //     BasicTokenType,
    // >,
    (),
> {
    let end_session_endpoint = global_state
        .get()
        .oidc_provider_metadata
        .additional_metadata()
        .end_session_endpoint
        .clone()
        .ok_or(anyhow!("None end_session_endpoint"))?;
    log::info!("end_session_endpoint={:?}", end_session_endpoint);
    let mut final_url = Url::parse(end_session_endpoint.as_str())?;
    final_url
        .query_pairs_mut()
        .append_pair("client_id", "thapo_ska_local_frontend")
        .append_pair(
            "post_logout_redirect_uri",
            "https://thapo-ska-local.thapo-local.org:9443/app/home",
        );
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

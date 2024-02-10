use std::sync::Arc;

use crate::modules::auth::auth_service;
use oauth2::{
    basic::{BasicErrorResponseType, BasicTokenType},
    EmptyExtraTokenFields, RevocationErrorResponseType, StandardErrorResponse,
    StandardRevocableToken, StandardTokenIntrospectionResponse, StandardTokenResponse,
};
use reqwest::Client;

use super::EnvConfig;
use openidconnect::{
    core::{
        CoreAuthDisplay, CoreAuthPrompt, CoreGenderClaim, CoreJsonWebKey, CoreJsonWebKeyType,
        CoreJsonWebKeyUse, CoreJweContentEncryptionAlgorithm, CoreJwsSigningAlgorithm,
    },
    EmptyAdditionalClaims, IdTokenFields,
};
#[derive(Clone, Debug)]
pub struct GlobalState {
    pub env_config: EnvConfig,
    pub api_client: Arc<Client>,
    pub oidc_provider_metadata: openidconnect::ProviderMetadata<
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
    >,
    pub oidc_client: openidconnect::Client<
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
}

impl GlobalState {
    pub async fn initialize_default() -> GlobalState {
        let env_config = EnvConfig::from_env();
        let api_client = Arc::new(Client::builder().build().unwrap_or_default());

        let (oidc_provider_metadata, oidc_client) =
            auth_service::create_oidc_client().await.unwrap();
        GlobalState {
            env_config,
            api_client,
            oidc_provider_metadata,
            oidc_client,
        }
    }
}

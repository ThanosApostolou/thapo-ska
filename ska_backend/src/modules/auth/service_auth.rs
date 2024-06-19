use axum::http::HeaderValue;
use hyper::HeaderMap;
use oauth2::{
    basic::{BasicErrorResponseType, BasicTokenType},
    AccessToken, Client, ClientId, ClientSecret, ExtraTokenFields, IntrospectionUrl,
    RevocationErrorResponseType, StandardErrorResponse, StandardRevocableToken,
    StandardTokenIntrospectionResponse, StandardTokenResponse, TokenIntrospectionResponse,
};

use serde::{Deserialize, Serialize};

use crate::{
    domain::repos::repo_users,
    modules::{
        auth::auth_models::UserDetails,
        db,
        error::{ErrorCode, ErrorResponse},
        global_state::{EnvConfig, GlobalState, SecretConfig},
    },
};
use std::{collections::HashSet, str::FromStr};

use super::auth_models::{AuthRoles, AuthTypes, AuthUser, UserAuthenticationDetails};

pub type MyAuthClient = Client<
    StandardErrorResponse<BasicErrorResponseType>,
    StandardTokenResponse<MyExtraTokenFields, BasicTokenType>,
    BasicTokenType,
    StandardTokenIntrospectionResponse<MyExtraTokenFields, BasicTokenType>,
    StandardRevocableToken,
    StandardErrorResponse<RevocationErrorResponseType>,
>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MyExtraTokenFields {
    pub realm_access: RealmAccess,
    pub email: String,
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
) -> anyhow::Result<MyAuthClient> {
    tracing::info!("auth_issuer_url={}", env_config.auth_issuer_url.clone());

    let auth_client: MyAuthClient = oauth2::Client::new(
        ClientId::new(env_config.auth_client_id.clone()),
        Some(ClientSecret::new(secret_config.auth_client_secret.clone())),
        oauth2::AuthUrl::new(env_config.auth_issuer_url.clone())?,
        Some(oauth2::TokenUrl::new(
            env_config.auth_issuer_url.clone() + "/protocol/openid-connect/token",
        )?),
    )
    .set_introspection_uri(IntrospectionUrl::new(
        env_config.auth_issuer_url.clone() + "/protocol/openid-connect/token/introspect",
    )?);

    Ok(auth_client)
}

async fn introspect(
    oidc_client: &MyAuthClient,
    access_token_str: String,
) -> anyhow::Result<StandardTokenIntrospectionResponse<MyExtraTokenFields, BasicTokenType>> {
    tracing::debug!("service_auth::introspect start");
    let access_token: AccessToken = AccessToken::new(access_token_str);
    let response: StandardTokenIntrospectionResponse<MyExtraTokenFields, BasicTokenType> =
        oidc_client
            .introspect(&access_token)?
            .request_async(my_async_http_client)
            .await?;
    tracing::info!("roles: {:?}", response.extra_fields().realm_access.roles);
    tracing::info!("active: {:?}", response.active());
    tracing::debug!("service_auth::introspect end");
    Ok(response)
}

async fn authenticate_user(
    global_state: &GlobalState,
    headers: &HeaderMap,
) -> anyhow::Result<UserAuthenticationDetails> {
    tracing::debug!("service_auth::authenticate_user start");
    let empty = HeaderValue::from_str("empty").unwrap();
    let header_authorization = headers.get("Authorization").unwrap_or(&empty).to_str()?;
    // tracing::info!("headers {}", header_authorization);
    let access_token_str = header_authorization
        .strip_prefix("Bearer ")
        .ok_or(anyhow::anyhow!("no bearer"))?;
    let response = introspect(&global_state.auth_client, access_token_str.to_string()).await?;
    if !response.active() {
        return Err(anyhow::anyhow!("token is not active"));
    }
    let ef: &MyExtraTokenFields = response.extra_fields();
    let sub = response.sub().ok_or(anyhow::anyhow!("no sub"))?.to_string();
    let username = response
        .username()
        .ok_or(anyhow::anyhow!("no username"))?
        .to_string();

    tracing::debug!("token roles:{:?}", &ef.realm_access.roles);
    let mut roles = HashSet::<AuthRoles>::new();
    for role_str in &ef.realm_access.roles {
        let role_result = AuthRoles::from_str(role_str);
        if let Ok(role) = role_result {
            roles.insert(role);
        }
    }

    let user_authentication_details: UserAuthenticationDetails = UserAuthenticationDetails {
        sub,
        username,
        email: ef.email.clone(),
        roles,
    };
    tracing::debug!("service_auth::authenticate_user end");
    Ok(user_authentication_details)
}

pub async fn perform_auth_user(
    global_state: &GlobalState,
    headers: &HeaderMap,
    auth_type: &AuthTypes,
) -> Result<AuthUser, ErrorResponse> {
    tracing::debug!("service_auth::perform_auth_user start");
    let txn: sea_orm::DatabaseTransaction = db::transaction_begin_read(global_state).await?;
    match auth_type {
        AuthTypes::Public => Ok(AuthUser::None),
        AuthTypes::Authentication => {
            let response_result = authenticate_user(global_state, headers).await;
            match response_result {
                Ok(user_authentication_details) => {
                    let authenticated = AuthUser::Authenticated(user_authentication_details);
                    db::transaction_commit(txn).await?;
                    tracing::debug!("service_auth::perform_auth_user Authentication ok");
                    return Ok(authenticated);
                }
                Err(e) => {
                    tracing::error!(
                        "service_auth::perform_auth_user Authentication error: {}",
                        e
                    );
                    Err(ErrorResponse {
                        error_code: ErrorCode::Unauthorized401,
                        is_unexpected_error: true,
                        packets: vec![],
                    })
                }
            }
        }
        AuthTypes::AuthorizationNoRoles => {
            let response_result = authenticate_user(global_state, headers).await;
            match response_result {
                Ok(user_authentication_details) => {
                    let user = repo_users::find_by_sub(
                        &global_state.db_connection,
                        &user_authentication_details.sub,
                    )
                    .await;

                    if let Ok(user) = user {
                        if let Some(user) = user {
                            let user_details = UserDetails {
                                user_authentication_details,
                                user_id: user.user_id,
                                last_login: user.last_login,
                            };
                            let authorized = AuthUser::Authorized(user_details);
                            db::transaction_commit(txn).await?;
                            tracing::debug!(
                                "service_auth::perform_auth_user AuthorizationNoRoles ok"
                            );
                            return Ok(authorized);
                        }
                    }
                    return Err(ErrorResponse {
                        error_code: ErrorCode::Unauthorized401,
                        is_unexpected_error: true,
                        packets: vec![],
                    });
                }
                Err(e) => {
                    tracing::error!(
                        "service_auth::perform_auth_user AuthorizationNoRoles error: {}",
                        e
                    );
                    return Err(ErrorResponse {
                        error_code: ErrorCode::Unauthorized401,
                        is_unexpected_error: true,
                        packets: vec![],
                    });
                }
            }
        }
        AuthTypes::Authorization(roles) => {
            let response_result = authenticate_user(global_state, headers).await;
            match response_result {
                Ok(user_authentication_details) => {
                    let mut has_roles = false;
                    for role in &user_authentication_details.roles {
                        if roles.contains(role) {
                            has_roles = true;
                            break;
                        }
                    }
                    if !has_roles {
                        return Err(ErrorResponse {
                            error_code: ErrorCode::Unauthorized401,
                            is_unexpected_error: true,
                            packets: vec![],
                        });
                    }

                    let user = repo_users::find_by_sub(
                        &global_state.db_connection,
                        &user_authentication_details.sub,
                    )
                    .await;

                    if let Ok(user) = user {
                        if let Some(user) = user {
                            let user_details = UserDetails {
                                user_authentication_details,
                                user_id: user.user_id,
                                last_login: user.last_login,
                            };
                            db::transaction_commit(txn).await?;
                            tracing::debug!("service_auth::perform_auth_user Authorization ok");
                            return Ok(AuthUser::Authorized(user_details));
                        }
                    }
                    return Err(ErrorResponse {
                        error_code: ErrorCode::Unauthorized401,
                        is_unexpected_error: true,
                        packets: vec![],
                    });
                }
                Err(e) => {
                    tracing::error!("service_auth::perform_auth_user Authorization error: {}", e);
                    return Err(ErrorResponse {
                        error_code: ErrorCode::Unauthorized401,
                        is_unexpected_error: true,
                        packets: vec![],
                    });
                }
            }
        }
    }
}

async fn my_async_http_client(
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

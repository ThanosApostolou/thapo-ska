use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use axum::routing::{get, post, MethodRouter};
use const_format::formatcp;

use crate::modules::auth::auth_models::AuthRoles;
use crate::server::handle_root;
use crate::server::route_api::route_assistant::route_ask_assistant_question::handle_ask_assistant_question;
use crate::server::route_api::route_assistant::{
    handle_create_chat, handle_fetch_assistant_options,
};
use crate::{
    modules::{auth::auth_models::AuthTypes, global_state::GlobalState},
    server::route_api::route_auth::route_app_login::handle_app_login,
};

#[derive(Clone, Debug)]
pub struct EndpointInfo {
    pub path: RoutePath,
    pub auth_type: AuthTypes,
    pub method_router: MethodRouter<Arc<GlobalState>>,
}
impl EndpointInfo {
    pub fn full_path_with_server_path(&self, server_path: String) -> String {
        server_path.clone() + self.path.parent_path + self.path.self_path
    }
}

// #[derive(
//     Clone, Debug, Serialize, Deserialize, AsRefStr, IntoStaticStr, EnumString, PartialEq, Eq, Hash,
// )]
// pub enum ApiEndpoints {
//     ApiAuthAppLogin,
// }

// impl ApiEndpoints {
//     pub fn get_endpoint_info(&self) -> EndpointInfo {
//         match self {
//             ApiEndpoints::ApiAuthAppLogin => EndpointInfo {
//                 self_path: "/app_login",
//                 nest_path: "/api/auth",
//                 auth_type: AuthTypes::Authentication,
//                 method_router: post(handle_app_login),
//             },
//         }
//     }
// }

#[derive(Clone, Debug)]
pub struct RoutePath {
    pub parent_path: &'static str,
    pub self_path: &'static str,
}

#[derive(Clone, Debug)]
pub struct Routes {
    pub route_api: RouteApi,
    pub root_endpoint: EndpointInfo,
}

impl Default for Routes {
    fn default() -> Self {
        Self::new()
    }
}

impl Routes {
    pub fn new() -> Routes {
        const API_STR: &str = "/api";
        const AUTH_STR: &str = "/auth";
        const ASSISTANT_STR: &str = "/assistant";

        let route_auth = RouteAuth {
            path: RoutePath {
                parent_path: API_STR,
                self_path: AUTH_STR,
            },
            endpoints: [EndpointInfo {
                path: RoutePath {
                    parent_path: formatcp!("{API_STR}{AUTH_STR}"),
                    self_path: "/app_login",
                },
                auth_type: AuthTypes::Authentication,
                method_router: post(handle_app_login),
            }],
        };

        let route_assistant = RouteAssistant {
            path: RoutePath {
                parent_path: API_STR,
                self_path: ASSISTANT_STR,
            },
            endpoints: [
                EndpointInfo {
                    path: RoutePath {
                        parent_path: formatcp!("{API_STR}{ASSISTANT_STR}"),
                        self_path: "/ask_assistant_question",
                    },
                    auth_type: AuthTypes::Authorization(HashSet::from([
                        AuthRoles::SkaAdmin,
                        AuthRoles::SkaUser,
                    ])),
                    method_router: get(handle_ask_assistant_question),
                },
                EndpointInfo {
                    path: RoutePath {
                        parent_path: formatcp!("{API_STR}{ASSISTANT_STR}"),
                        self_path: "/fetch_assistant_options",
                    },
                    auth_type: AuthTypes::Authorization(HashSet::from([
                        AuthRoles::SkaAdmin,
                        AuthRoles::SkaUser,
                    ])),
                    method_router: get(handle_fetch_assistant_options),
                },
                EndpointInfo {
                    path: RoutePath {
                        parent_path: formatcp!("{API_STR}{ASSISTANT_STR}"),
                        self_path: "/create_chat",
                    },
                    auth_type: AuthTypes::Authorization(HashSet::from([
                        AuthRoles::SkaAdmin,
                        AuthRoles::SkaUser,
                    ])),
                    method_router: post(handle_create_chat),
                },
            ],
        };

        Routes {
            root_endpoint: EndpointInfo {
                path: RoutePath {
                    parent_path: "",
                    self_path: "",
                },
                auth_type: AuthTypes::Public,
                method_router: get(handle_root),
            },
            route_api: RouteApi {
                path: RoutePath {
                    parent_path: "",
                    self_path: API_STR,
                },
                route_auth,
                route_assistant,
            },
        }
    }

    pub fn routes_auth_types_map(&self, server_path: String) -> HashMap<String, AuthTypes> {
        let mut map: HashMap<String, AuthTypes> = HashMap::new();
        map.insert(
            self.root_endpoint
                .full_path_with_server_path(server_path.clone()),
            self.root_endpoint.auth_type.clone(),
        );

        for endpoint in &self.route_api.route_auth.endpoints {
            map.insert(
                endpoint.full_path_with_server_path(server_path.clone()),
                endpoint.auth_type.clone(),
            );
        }
        for endpoint in &self.route_api.route_assistant.endpoints {
            map.insert(
                endpoint.full_path_with_server_path(server_path.clone()),
                endpoint.auth_type.clone(),
            );
        }
        // HashMap::from([
        //     (
        //         self.route_api,
        //         AuthTypes::Authentication,
        //     ),
        //     (
        //         formatcp!(
        //             "{}{}{}{}",
        //             env_config.server_path,
        //             PATH_API,
        //             PATH_AUTH,
        //             PATH_APP_LOGIN
        //         ),
        //         AuthTypes::Authentication,
        //     ),
        // ])
        map
    }
}

#[derive(Clone, Debug)]
pub struct RouteApi {
    pub path: RoutePath,
    pub route_auth: RouteAuth,
    pub route_assistant: RouteAssistant,
}

#[derive(Clone, Debug)]
pub struct RouteAuth {
    pub path: RoutePath,
    pub endpoints: [EndpointInfo; 1],
}

#[derive(Clone, Debug)]
pub struct RouteAssistant {
    pub path: RoutePath,
    pub endpoints: [EndpointInfo; 3],
}

// const fn concat(str1: &'static str)
// #[derive(Debug)]
// pub struct Routes {
//     route_api: RouteApi,
// }

// #[derive(Debug)]
// pub struct RouteApi {
//     self_path: &'static str,
//     route_auth: Vec<EndpointInfo>,
//     route_assistant: Vec<EndpointInfo>,
// }
// #[derive(Debug)]
// pub struct RouteAssistant {
//     self_path: &'static str,
//     endpoints: Vec<EndpointInfo>,
// }

// impl Routes {
//     pub fn get_routes(server_path: String) -> Routes {
//         let api_str = "/api";
//         return Routes {
//             route_api: RouteApi {
//                 self_path: api_str,
//                 route_auth: vec![EndpointInfo {
//                     parent_path: server_path + api_str,
//                     nest_path: "/auth",
//                     self_path: "/app_login",
//                     auth_type: AuthTypes::Authentication,
//                     method_router: post(handle_app_login),
//                 }],
//             },
//         };
//     }
// }

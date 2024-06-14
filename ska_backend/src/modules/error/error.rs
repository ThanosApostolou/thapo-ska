use std::fmt;

use axum::http;
use serde::{Deserialize, Serialize};
use strum_macros::{AsRefStr, IntoStaticStr};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub error_code: ErrorCode,
    pub is_unexpected_error: bool,
    pub packets: Vec<ErrorPacket>,
}

impl ErrorResponse {
    pub fn packets(&self) -> &Vec<ErrorPacket> {
        &self.packets
    }

    pub fn new(
        error_code: ErrorCode,
        is_unexpected_error: bool,
        packets: Vec<ErrorPacket>,
    ) -> ErrorResponse {
        ErrorResponse {
            error_code,
            is_unexpected_error,
            packets,
        }
    }

    pub fn new_standard(
        message: String,
        is_unexpected_error: bool,
        is_frontend: bool,
    ) -> ErrorResponse {
        ErrorResponse {
            error_code: ErrorCode::UnprocessableEntity422,
            is_unexpected_error,
            packets: vec![ErrorPacket {
                message: {
                    if is_frontend {
                        message.clone()
                    } else {
                        "".to_string()
                    }
                },
                backend_message: message,
            }],
        }
    }
}

impl fmt::Display for ErrorResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let packets_str: String = self
            .packets
            .iter()
            .map(|packet| packet.backend_message.clone())
            .collect::<Vec<String>>()
            .join(",");
        let error_packets_str = format!("[{}]", packets_str);
        write!(
            f,
            "status_code={}, is_unexpected_error={}, packets={}",
            self.error_code, self.is_unexpected_error, error_packets_str
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErrorPacket {
    pub message: String,
    pub backend_message: String,
}

impl ErrorPacket {
    pub fn message(&self) -> &String {
        &self.message
    }
    pub fn backend_message(&self) -> &String {
        &self.backend_message
    }

    pub fn new(error_packet: ErrorPacket) -> ErrorPacket {
        let message = error_packet.message().clone();
        let backend_message = error_packet.backend_message().clone();
        ErrorPacket {
            message,
            backend_message,
        }
    }

    pub fn new_backend(backend_message: &str) -> ErrorPacket {
        let message = "".to_string();
        let backend_message = backend_message.to_string();
        ErrorPacket {
            message,
            backend_message,
        }
    }
}

impl fmt::Display for ErrorPacket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "message={}, backend_message={}",
            self.message, self.backend_message
        )
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, AsRefStr, IntoStaticStr, PartialEq)]
pub enum ErrorCode {
    BadRequest400,
    Unauthorized401,
    Forbidden403,
    UnprocessableEntity422,
    InternalServerError500,
}

impl ErrorCode {
    pub fn into_status_code(&self) -> http::StatusCode {
        match self {
            ErrorCode::BadRequest400 => http::StatusCode::BAD_REQUEST,
            ErrorCode::Unauthorized401 => http::StatusCode::UNAUTHORIZED,
            ErrorCode::Forbidden403 => http::StatusCode::FORBIDDEN,
            ErrorCode::UnprocessableEntity422 => http::StatusCode::UNPROCESSABLE_ENTITY,
            ErrorCode::InternalServerError500 => http::StatusCode::INTERNAL_SERVER_ERROR,
        }
    }

    pub fn into_u16(&self) -> u16 {
        match self {
            ErrorCode::BadRequest400 => 400,
            ErrorCode::Unauthorized401 => 401,
            ErrorCode::Forbidden403 => 403,
            ErrorCode::UnprocessableEntity422 => 422,
            ErrorCode::InternalServerError500 => 500,
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_ref())
    }
}

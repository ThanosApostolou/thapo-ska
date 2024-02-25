use std::fmt;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ErrorResponse {
    pub status_code: u16,
    pub is_unexpected_error: bool,
    pub packets: Vec<ErrorPacket>,
}

impl ErrorResponse {
    pub fn packets(&self) -> &Vec<ErrorPacket> {
        &self.packets
    }

    pub fn new(
        status_code: u16,
        is_unexpected_error: bool,
        packets: Vec<ErrorPacket>,
    ) -> ErrorResponse {
        ErrorResponse {
            status_code,
            is_unexpected_error,
            packets,
        }
    }
}

impl fmt::Display for ErrorResponse {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "status_code={}, is_unexpected_error={}, #packets={}",
            self.status_code,
            self.is_unexpected_error,
            self.packets.len()
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

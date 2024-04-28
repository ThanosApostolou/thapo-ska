use std::fmt;

use serde::{Deserialize, Serialize};

use super::{ErrorPacket, ErrorResponse};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DtoErrorResponse {
    pub status_code: u16,
    pub is_unexpected_error: bool,
    pub packets: Vec<DtoErrorPacket>,
}

impl DtoErrorResponse {
    pub fn packets(&self) -> &Vec<DtoErrorPacket> {
        &self.packets
    }

    pub fn new(
        status_code: u16,
        is_unexpected_error: bool,
        packets: Vec<DtoErrorPacket>,
    ) -> DtoErrorResponse {
        DtoErrorResponse {
            status_code,
            is_unexpected_error,
            packets,
        }
    }

    pub fn from_error_response(error_response: ErrorResponse) -> DtoErrorResponse {
        DtoErrorResponse {
            status_code: error_response.error_code.into_u16(),
            is_unexpected_error: error_response.is_unexpected_error,
            packets: DtoErrorPacket::vec_from_error_packets(error_response.packets),
        }
    }
}

impl fmt::Display for DtoErrorResponse {
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
pub struct DtoErrorPacket {
    pub message: String,
}

impl DtoErrorPacket {
    pub fn message(&self) -> &String {
        &self.message
    }

    pub fn new(message: String) -> DtoErrorPacket {
        DtoErrorPacket { message }
    }

    pub fn from_error_packet(error_packet: ErrorPacket) -> DtoErrorPacket {
        DtoErrorPacket {
            message: error_packet.message,
        }
    }

    pub fn vec_from_error_packets(error_packets: Vec<ErrorPacket>) -> Vec<DtoErrorPacket> {
        error_packets
            .into_iter()
            .map(|error_packet| DtoErrorPacket {
                message: error_packet.message,
            })
            .collect()
    }
}

impl fmt::Display for DtoErrorPacket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "message={}", self.message)
    }
}

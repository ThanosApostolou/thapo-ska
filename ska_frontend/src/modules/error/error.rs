use std::fmt::{self};

use serde::{Deserialize, Serialize};

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
}

impl fmt::Display for DtoErrorPacket {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "message={}", self.message)
    }
}

use crate::modules::error::ErrorPacket;

pub fn syntax_message_body(message_body: &Option<&String>) -> Result<(), ErrorPacket> {
    if let Some(message_body) = message_body {
        let len = message_body.trim().len();
        if len < 3 || len > 384 {
            let message = "message size must be between 3 and 384".to_string();
            return Err(ErrorPacket {
                message: message.clone(),
                backend_message: message,
            });
        }
    }
    return Ok(());
}

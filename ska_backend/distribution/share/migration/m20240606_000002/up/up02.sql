CREATE TABLE IF NOT EXISTS thapo_ska_schema.chat_message(
    chat_message_id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    chat_id_fk BIGINT NOT NULL,
    message_type VARCHAR(20) NOT NULL,
    message_body VARCHAR(1000) NOT NULL,
    context jsonb NOT NULL,
    created_at timestamp without time zone NOT NULL,
    FOREIGN KEY (chat_id_fk) REFERENCES thapo_ska_schema.user_chat (chat_id)
);
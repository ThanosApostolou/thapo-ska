CREATE TABLE IF NOT EXISTS thapo_ska_schema.user_chat(
    chat_id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
    user_id_fk BIGINT NOT NULL,
    chat_name VARCHAR(200) NOT NULL,
    prompt VARCHAR(512) NULL,
    temperature NUMERIC(3, 2) NULL CHECK (temperature >= 0 AND temperature <= 1),
    top_p NUMERIC(3, 2) NULL CHECK (top_p >= 0 AND top_p <= 1),
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    FOREIGN KEY (user_id_fk) REFERENCES thapo_ska_schema.users (user_id)
);
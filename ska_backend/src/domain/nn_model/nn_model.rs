use serde::{Deserialize, Serialize};
use strum_macros::{AsRefStr, Display, EnumString, IntoStaticStr};

#[derive(Debug, Clone)]
pub enum LlmModelTypeEnum {
    CtransformersLlama,
    LlamaCpp,
    HuggingFace,
    Skalm,
}

impl LlmModelTypeEnum {
    pub fn get_value(&self) -> &str {
        match self {
            LlmModelTypeEnum::CtransformersLlama => "ctransformers_llama",
            LlmModelTypeEnum::LlamaCpp => "llamacpp",
            LlmModelTypeEnum::HuggingFace => "huggingface",
            LlmModelTypeEnum::Skalm => "skalm",
        }
    }
}

#[derive(Debug, Clone)]
pub enum NnModelEnum {
    AllMiniLML6,
    Llama27BChat,
    Llama38BInstruct,
    // TinyLlama1_1BChat,
    // Opt350M,
    // Gpt2,
}
impl NnModelEnum {
    pub fn get_data(&self) -> NnModelData {
        match self {
            NnModelEnum::AllMiniLML6 => NnModelData {
                name: "all-MiniLM-L6-v2".to_string(),
                repo_id: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
                rel_path: "all-MiniLM-L6-v2".to_string(),
                model_path: "all-MiniLM-L6-v2".to_string(),
                revision: "e4ce9877abf3edfe10b0d82785e83bdcb973e22e".to_string(),
                allow_patterns: "*".to_string(),
                ignore_patterns: "".to_string(),
                model_type: NnModelType::ModelEmbedding,
                default_prompt: "".to_string(),
                llm_model_type: None,
            },
            NnModelEnum::Llama27BChat => NnModelData {
                name: "llama2-7B".to_string(),
                repo_id: "TheBloke/Llama-2-7B-Chat-GGUF".to_string(),
                rel_path: "Llama-2-7B-Chat-GGUF".to_string(),
                model_path: "Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q2_K.gguf".to_string(),
                revision: "191239b3e26b2882fb562ffccdd1cf0f65402adb".to_string(),
                allow_patterns: "*.json,*.txt,*.md,*.model,llama-2-7b-chat.Q2_K.gguf".to_string(),
                ignore_patterns: "*.bin,*.h5,*.msgpack,*.ot, *.safetensors".to_string(),
                model_type: NnModelType::ModelLlm,
                default_prompt: "<s>[INST] <<SYS>>
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<</SYS>>
Context: {context}
Question: {question}
Answer: [/INST]".to_string(),
                llm_model_type: Some(LlmModelTypeEnum::LlamaCpp),
            },
            NnModelEnum::Llama38BInstruct => NnModelData {
                name: "llama3-8B".to_string(),
                repo_id: "SanctumAI/Meta-Llama-3-8B-Instruct-GGUF".to_string(),
                rel_path: "Meta-Llama-3-8B-Instruct-GGUF".to_string(),
                model_path: "Meta-Llama-3-8B-Instruct-GGUF/meta-llama-3-8b-instruct.Q2_K.gguf".to_string(),
                revision: "f688151a21ac4496648f183682ac25772b110658".to_string(),
                allow_patterns: "*.json,*.txt,*.md,*.model,meta-llama-3-8b-instruct.Q2_K.gguf".to_string(),
                ignore_patterns: "*.bin,*.h5,*.msgpack,*.ot, *.safetensors".to_string(),
                model_type: NnModelType::ModelLlm,
                default_prompt: "<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.<|eot_id|><|start_header_id|>user<|end_header_id|>
Context: {context}. Question: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>".to_string(),
                llm_model_type: Some(LlmModelTypeEnum::LlamaCpp),
            },
            // NnModelEnum::TinyLlama1_1BChat => NnModelData {
            //     name: "tinyllama1_1B".to_string(),
            //     repo_id: "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T".to_string(),
            //     rel_path: "TinyLlama-1.1B-intermediate-step-1431k-3T".to_string(),
            //     model_path: "TinyLlama-1.1B-intermediate-step-1431k-3T".to_string(),
            //     revision: "036fa4651240b9a1487f709833b9e4b96b4c1574".to_string(),
            //     allow_patterns: "*".to_string(),
            //     ignore_patterns: "*.bin".to_string(),
            //     model_type: NnModelType::ModelLlm,
            //     default_prompt: "".to_string(),
            //     llm_model_type: Some(LlmModelTypeEnum::HuggingFace),
            // },
            // NnModelEnum::Opt350M => NnModelData {
            //     name: "opt350m".to_string(),
            //     repo_id: "facebook/opt-350m".to_string(),
            //     rel_path: "opt-350m".to_string(),
            //     model_path: "opt-350m".to_string(),
            //     revision: "08ab08cc4b72ff5593870b5d527cf4230323703c".to_string(),
            //     allow_patterns: "*.json,*.txt,*.md,*.bin,*.model".to_string(),
            //     ignore_patterns: "*.safetensors,*.h5,*.msgpack,*.ot".to_string(),
            //     model_type: NnModelType::ModelLlm,
            //     default_prompt: "".to_string(),
            //     llm_model_type: Some(LlmModelTypeEnum::HuggingFace),
            // },

            // NnModelEnum::Gpt2 => NnModelData {
            //     name: "gpt2".to_string(),
            //     repo_id: "openai-community/gpt2".to_string(),
            //     rel_path: "gpt2".to_string(),
            //     model_path: "gpt2".to_string(),
            //     revision: "607a30d783dfa663caf39e06633721c8d4cfcd7e".to_string(),
            //     allow_patterns: "*.json,*.txt,*.md,*.safetensors,*.model".to_string(),
            //     ignore_patterns: "*.bin,*.h5,*.msgpack,*.ot".to_string(),
            //     model_type: NnModelType::ModelLlm,
            //     default_prompt: "".to_string(),
            //     llm_model_type: Some(LlmModelTypeEnum::HuggingFace),
            // },
        }
    }

    pub fn get_data_list() -> Vec<NnModelData> {
        Vec::<NnModelData>::from([
            NnModelEnum::AllMiniLML6.get_data(),
            NnModelEnum::Llama27BChat.get_data(),
            NnModelEnum::Llama38BInstruct.get_data(),
            // NnModelEnum::TinyLlama1_1BChat.get_data(),
            // NnModelEnum::Opt350M.get_data(),
            // NnModelEnum::Gpt2.get_data(),
        ])
    }
}

#[derive(Debug, Clone)]
pub struct NnModelData {
    pub name: String,
    pub repo_id: String,
    pub rel_path: String,
    pub model_path: String,
    pub revision: String,
    pub allow_patterns: String,
    pub ignore_patterns: String,
    pub model_type: NnModelType,
    pub default_prompt: String,
    pub llm_model_type: Option<LlmModelTypeEnum>,
}

impl NnModelData {
    pub fn into_tuple(&self) -> (String, String, String, String) {
        (
            self.repo_id.clone(),
            self.rel_path.clone(),
            self.revision.clone(),
            self.allow_patterns.clone(),
        )
    }

    pub fn get_skalm_data() -> NnModelData {
        NnModelData {
            name: "skalm".to_string(),
            repo_id: "".to_string(),
            rel_path: "skalm".to_string(),
            model_path: "skalm".to_string(),
            revision: "".to_string(),
            allow_patterns: "".to_string(),
            ignore_patterns: "".to_string(),
            model_type: NnModelType::ModelLlm,
            default_prompt: "".to_string(),
            llm_model_type: Some(LlmModelTypeEnum::Skalm),
        }
    }
}

#[derive(
    Clone,
    Debug,
    Serialize,
    Deserialize,
    AsRefStr,
    IntoStaticStr,
    EnumString,
    Display,
    PartialEq,
    Eq,
    Hash,
)]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum NnModelType {
    #[serde(rename = "MODEL_EMBEDDING")]
    ModelEmbedding,
    #[serde(rename = "MODEL_LLM")]
    ModelLlm,
}

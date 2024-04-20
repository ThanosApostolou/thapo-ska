#EMB_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
LL_MODEL = "../models/llm/llama-2-7b-chat/llama-2-7b-chat.ggmlv3.q8_0.bin"

"""model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
LL_MODEL = AutoModelForCausalLM.from_pretrained(model_id)"""
"""LL_MODEL = OpenLLM(
                model_name="dolly-v2",
                model_id="databricks/dolly-v2-3b",
                temperature=0.94,
                repetition_penalty=1.2,
            )"""

MAX_TOKENS = 512
TEMPERATURE = 0.01
CONTEXT_LEN = 4096

# prepare the template we will use when prompting the AI
PROMPT_TEMPLATE = """<s>[INST] <<SYS>>
Task: Your task is to assist as a helper-agent.
Instruction:
1. Extract relevant information and answer using only the context provided .
2. Only respond if the user's question aligns with the following context provided.
3. If the user's question is irrelevant to the context, refrain from providing an answer, even if you possess the knowledge.
4. If the question does not align with the following context, display the message: "I don't know the answer for this task. Please try again."
5. Never refer the context as a word in the response
Context: {context}
<</SYS>>

User: {question} [/INST]"""

VS_INDEX_NAME = "llm_rag"
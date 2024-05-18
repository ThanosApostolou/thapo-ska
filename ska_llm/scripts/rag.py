import json
from typing import Any, Self
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
# from langchain_community.document_loaders.epub import UnstructuredEPubLoader
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain_community.document_loaders.xml import UnstructuredXMLLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from langchain_community.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate
# from langchain.chains.retrieval_qa.base import RetrievalQA, BaseRetrievalQA
from langchain_core.callbacks import CallbackManager, StdOutCallbackHandler
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from scripts import constants

class InvokeOutput:
    def __init__(self):
        self.context: list[Document] = []
        self.question: str = ''
        self.answer: str = ''

    @classmethod
    def from_output_dict(cls, output_dict: dict[Any, Any]) -> Self:
        invoke_output = cls()
        invoke_output.context = output_dict['context']
        assert isinstance(invoke_output.context, list)
        assert isinstance(invoke_output.question, str)
        assert isinstance(invoke_output.answer, str)
        invoke_output.question = output_dict['question']
        invoke_output.answer = output_dict['answer']
        return invoke_output



class DocumentDto(object):
    def __init__(self, page_content: str, metadata: dict[Any, Any]):
        self.page_content: str = page_content
        self.metadata: dict[Any, Any] = metadata

    def __repr__(self) -> str:
        return self.__dict__.__repr__()

    def to_json_obj(self) -> dict[Any, Any]:
        return self.__dict__


    @classmethod
    def from_document(cls, document: Document) -> Self:
        invoke_output = cls(document.page_content, document.metadata)
        return invoke_output

class InvokeOutputDto(object):
    def __init__(self, context: list[DocumentDto], question: str, answer: str):
        self.context: list[DocumentDto] = context
        self.question: str = question
        self.answer: str = answer

    def __repr__(self) -> str:
        return self.__dict__.__repr__()

    def to_json_obj(self) -> dict:
        return {
            "context": [x.__dict__ for x in self.context],
            "question": self.question,
            "answer": self.answer
        }


    @classmethod
    def from_invoke_output(cls, invoke_output: InvokeOutput) -> Self:
        context_dtos = list(map(DocumentDto.from_document, invoke_output.context))
        invoke_output_dto = cls(context_dtos, invoke_output.question, invoke_output.answer)
        return invoke_output_dto


def get_embeddings(embedding_model_path: str):
    print("get_embeddings start")
    emb =  HuggingFaceBgeEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs = {'normalize_embeddings': True}
    )
    print("get_embeddings end")
    return emb


def read_docs(data_path: str) -> list[Document]:
    print("DirectoryLoader txt")
    txt_loader = DirectoryLoader(data_path, glob="**/*.txt", loader_cls=TextLoader, silent_errors=True, show_progress=True, use_multithreading=True)
    txt_docs = txt_loader.load()

    print("DirectoryLoader md")
    md_loader = DirectoryLoader(data_path, glob="**/*.md", loader_cls=TextLoader, silent_errors=True, show_progress=True, use_multithreading=True)
    md_docs = md_loader.load()

    print("DirectoryLoader pdf")
    pdf_loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=UnstructuredPDFLoader, silent_errors=True, show_progress=True, use_multithreading=True)
    pdf_docs = pdf_loader.load()

    # print("DirectoryLoader epub")
    # epub_loader = DirectoryLoader(data_path, glob="**/*.epub", loader_cls=UnstructuredEPubLoader, silent_errors=True, show_progress=True, use_multithreading=True)
    # epub_docs = epub_loader.load()

    print("DirectoryLoader html")
    html_loader = DirectoryLoader(data_path, glob="**/*.html", loader_cls=UnstructuredHTMLLoader, silent_errors=True, show_progress=True, use_multithreading=True)
    html_docs = html_loader.load()

    print("DirectoryLoader xml")
    xml_loader = DirectoryLoader(data_path, glob="**/*.xml", loader_cls=UnstructuredXMLLoader, silent_errors=True, show_progress=True, use_multithreading=True)
    xml_docs = xml_loader.load()

    docs = txt_docs + md_docs + pdf_docs + html_docs + xml_docs
    print(f"len(docs): {len(docs)}")
    return docs




def prepare(data_path: str, vector_store_path: str, embedding_model_path: str):
    """
    Retrieval Augmented Generation (RAG) prepare
    """

    docs = read_docs(data_path)

    # transformation: split the documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=128,
        chunk_overlap=8
    )

    print("splitter.split_documents")
    texts = splitter.split_documents(docs)

    # create the embeddings and store to database
    print("get_embeddings()")
    embeddings = get_embeddings(embedding_model_path)

    # create and save the local database
    print("FAISS.from_documents")
    db = FAISS.from_documents(texts, embeddings)
    print("FAISS.save_local")
    db.save_local(vector_store_path, constants.VS_INDEX_NAME)


def create_llm(llm_model_path: str, model_type: str):
    context_length = 384
    max_tokens = 128
    top_p = 0.95
    temperature = 0.8
    batch_size = 8
    last_n_tokens = 32
    repetition_penalty = 1.1

    if model_type == 'llamacpp':
        # Callbacks support token-wise streaming
        callback_manager = CallbackManager([StdOutCallbackHandler()])
        llm = LlamaCpp(
            model_path=llm_model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            last_n_tokens_size=last_n_tokens,
            top_p=top_p,
            callback_manager=callback_manager,
            verbose=True,  # Verbose is required to pass to the callback manager
            client=None,
            n_ctx=context_length,
            n_parts=-1,
            seed=-1,
            f16_kv=True,
            logits_all=False,
            vocab_only=False,
            use_mlock=False,
            n_threads=None,
            n_batch=batch_size,
            n_gpu_layers=None,
            suffix=None,
            logprobs=None,
            repeat_penalty=repetition_penalty
        )
        return llm
    elif model_type == 'huggingface':
        tokenizer = AutoTokenizer.from_pretrained(llm_model_path)
        model = AutoModelForCausalLM.from_pretrained(llm_model_path, device_map='cpu')
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=context_length)
        llm = HuggingFacePipeline(pipeline=pipe)

        # llm = HuggingFacePipeline.from_model_id(
        #     model_id=llm_model_path,
        #     task="text-generation",
        #     pipeline_kwargs={"max_new_tokens": max_new_tokens},
        # )
        return llm
    else:
        raise Exception(f"unsuported model_type {model_type}")

def create_chain(vector_store_path: str, embedding_model_path: str, llm_model_path: str, prompt_template: str, model_type: str):
    """
    Retrieval Augmented Generation (RAG) prepare
    """

    # load the language model
    # TODO check https://github.com/marella/ctransformers
    llm = create_llm(llm_model_path, model_type)

    # load the interpreted information from the local database
    embeddings = get_embeddings(embedding_model_path)
    db = FAISS.load_local(vector_store_path, embeddings, constants.VS_INDEX_NAME, allow_dangerous_deserialization=True)

    # prepare a version of the llm pre-loaded with the local content
    retriever = db.as_retriever(search_kwargs={'k': 9})
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=['context', 'question']
    )
    output_parser = StrOutputParser()
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     chain_type='stuff',
    #     retriever=retriever,
    #     return_source_documents=True,
    #     chain_type_kwargs={
    #         'verbose': True,
    #         'prompt': prompt}
    # )
    # return qa_chain


    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain_from_docs = (
        RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    rag_chain_with_source = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    ).assign(answer=rag_chain_from_docs)

    return rag_chain_with_source


def invoke(vector_store_path: str, embedding_model_path: str, llm_model_path: str, prompt_template: str, question: str, model_type: str) -> InvokeOutput:
    qa_chain = create_chain(vector_store_path, embedding_model_path, llm_model_path, prompt_template, model_type)
    output = qa_chain.invoke(question)
    invoke_output = InvokeOutput.from_output_dict(output)
    return invoke_output

from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.pdf import UnstructuredPDFLoader
# from langchain_community.document_loaders.epub import UnstructuredEPubLoader
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain_community.document_loaders.xml import UnstructuredXMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS

from langchain_community.llms.ctransformers import CTransformers
from langchain_community.llms.llamacpp import LlamaCpp
from langchain.prompts import PromptTemplate
from langchain.chains.retrieval_qa.base import RetrievalQA


from ska_llm.scripts import constants

def get_embeddings(embedding_model_path: str):
    return HuggingFaceEmbeddings(
        model_name=embedding_model_path,
        model_kwargs={'device': 'cpu'}
    )


def prepare(data_path: str, vector_store_path: str, embedding_model_path: str):
    """
    Retrieval Augmented Generation (RAG) prepare
    """

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

    # transformation: split the documents into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )

    print("splitter.split_documents")
    texts = splitter.split_documents(docs)

    # create the embeddings and store to database
    embeddings = get_embeddings(embedding_model_path)

    # create and save the local database
    print("FAISS.from_documents")
    db = FAISS.from_documents(texts, embeddings)
    print("FAISS.save_local")
    db.save_local(vector_store_path, constants.VS_INDEX_NAME)


def fit(vector_store_path: str, embedding_model_path: str):
    """
    Retrieval Augmented Generation (RAG) prepare
    """

    # load the language model
    # TODO check https://github.com/marella/ctransformers
    llm = CTransformers(
        model=constants.LL_MODEL,
        model_type='llama',
        config={
             'context_length': constants.CONTEXT_LEN,
        },
        client=None
    )

    # load the interpreted information from the local database
    embeddings = get_embeddings(embedding_model_path)
    db = FAISS.load_local(vector_store_path, embeddings, constants.VS_INDEX_NAME)

    # prepare a version of the llm pre-loaded with the local content
    retriever = db.as_retriever(search_kwargs={'k': 9})
    prompt = PromptTemplate(
        template=constants.PROMPT_TEMPLATE,
        input_variables=['context', 'question']
    )
    qa_llm = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            'verbose': True,
            'prompt': prompt}

    )
    return qa_llm

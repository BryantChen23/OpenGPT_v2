import streamlit as st
import os
import pandas as pd

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredPowerPointLoader,
    UnstructuredExcelLoader,
)
from langchain.chains.query_constructor.base import AttributeInfo
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain.prompts.prompt import PromptTemplate
from langchain.chains.query_constructor.base import (
    StructuredQueryOutputParser,
    get_query_constructor_prompt,
)
from langchain.retrievers.self_query.chroma import ChromaTranslator
from langchain.retrievers.self_query.base import SelfQueryRetriever

# from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# webpage title setting
st.set_page_config(page_title="OpenGPT Prototype", page_icon="📎")
st.subheader("Document Chat")


def get_vectordb(assistant_id):
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    model = HuggingFaceEmbeddings(model_name=model_name)

    vectordb = Chroma(
        persist_directory=f"./Chroma/{assistant_id}",
        collection_name=assistant_id,
        embedding_function=model,
    )
    return vectordb


def get_filename(vectordb):
    filenames_dul = []
    for metadata in vectordb.get()["metadatas"]:
        filenames_dul.append(metadata["filename"])
        filenames = list(set(filenames_dul))
    return filenames


def get_query_retriever(vectordb):

    document_content_description = "These are the information about the conversation."

    metadata_field_info = [
        AttributeInfo(
            name="source",
            description="The source location of the document.",
            type="string",
        ),
        AttributeInfo(
            name="filename", description="The name of the document", type="string"
        ),
    ]

    # LLM - OpenAI
    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-15-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-4-assistant",
    )

    query_retriever = SelfQueryRetriever.from_llm(
        llm,
        vectordb,
        document_contents=document_content_description,
        metadata_field_info=metadata_field_info,
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 8},
    )

    return query_retriever


def get_normal_retriever(vectordb):

    normal_retreiver = vectordb.as_retriever(
        search_type="mmr", search_kwargs={"k": 6, "fetch_k": 10}
    )

    return normal_retreiver


# add filename into filtered_docs page_content
def doc_content_add_filename(docs):
    for doc in docs:
        doc.page_content = (
            "File Name:" + doc.metadata["filename"] + "\n" + doc.page_content
        )
    return docs


# Memory initial
document_chat_msgs = StreamlitChatMessageHistory(key="doc_qa_messages")

# 新對話視
if st.sidebar.button(label="New Chat", type="secondary"):
    document_chat_msgs.clear()

ass_df = pd.read_csv("./opengpt_app_data/assistant_admin.csv")
assistant = st.sidebar.selectbox(
    label="Assistant Options", options=ass_df["assistant_name"], index=None
)


if not assistant == None:

    assistant_id = ass_df[ass_df["assistant_name"] == assistant]["assistant_id"].values[
        0
    ]
    vectordb = get_vectordb(assistant_id)

    query_retriever = get_query_retriever(vectordb)
    # query_retriever = get_normal_retriever(vectordb)

    # LLM Conversation Model
    # llm = ChatOllama(model="phi3")
    llm = AzureChatOpenAI(
        api_key=os.getenv("AZURE_OPENAI_KEY"),
        api_version="2024-02-15-preview",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-4-assistant",
    )

    # Memory
    memory = ConversationBufferWindowMemory(
        return_messages=True,
        input_key="query",
        memory_key="chat_history",
        chat_memory=document_chat_msgs,
    )

    prompt_template = """
    Use the following pieces of context and chat history to answer the question at the end.
    When reading documents, filling out documents, or replying, please respond based on the file's content without translating or creating any answers.

    Context: {context}

    Chat history:{chat_history}

    Question: {query}
    """

    prompt = PromptTemplate(
        input_variables=["chat_history", "query", "context"],
        template=prompt_template,
    )

    # Chain
    chain = load_qa_chain(
        llm=llm,
        chain_type="stuff",
        memory=memory,
        prompt=prompt,
        verbose=True,
    )

    # Initialize st_chat history and create message container
    if len(document_chat_msgs.messages) == 0:
        document_chat_msgs.add_ai_message("How can I help you today?")

    # Display history message
    avatars = {"human": "user", "ai": "assistant"}
    for msg in document_chat_msgs.messages:
        st.chat_message(avatars[msg.type]).write(msg.content)

    # User/AI Conversation
    if query := st.chat_input(placeholder="Ask me anything!"):
        st.chat_message("user").write(query)
        with st.chat_message("assistant"):
            filtered_docs = query_retriever.invoke(query)
            input_docs = doc_content_add_filename(filtered_docs)
            response = chain({"query": query, "input_documents": input_docs})
            st.markdown(response["output_text"])

else:
    st.info("Please choose assistant to continue.")
    document_chat_msgs.clear()
    st.stop()

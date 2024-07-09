import streamlit as st
import pandas as pd
import os
import tempfile

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


def check_and_create(directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)

    files = os.listdir(directory)

    if not filename in files:
        filepath = os.path.join(directory, filename)
        df = pd.DataFrame({"assistant_name": [], "document": []})
        df.to_csv(filepath, index=False, encoding="utf-8-sig")


def get_file_type(filename):
    _, file_type = os.path.splitext(filename)
    return file_type.lower()


def file_load(uploaded_files):
    """read document"""
    docs = []
    temp_dir = tempfile.TemporaryDirectory()

    loader_map = {
        ".doc": Docx2txtLoader,
        ".docx": Docx2txtLoader,
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader,
        ".xlsx": UnstructuredExcelLoader,
        ".pptx": UnstructuredPowerPointLoader,
    }

    for file in uploaded_files:
        file_type = get_file_type(file.name)
        if file_type in loader_map:
            temp_path = os.path.join(temp_dir.name, file.name)
            with open(temp_path, "wb") as temp_file:
                temp_file.write(file.getvalue())
            loader = loader_map[file_type]
            docs.extend(loader(temp_path).load())

    for doc in docs:
        filename = os.path.splitext(os.path.basename(doc.metadata["source"]))[0]
        doc.metadata["filename"] = filename

    return docs


def file_splitter(docs):
    """document split"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
    splits = splitter.split_documents(docs)

    return splits


def embedding_to_vector(document_splits, docstore_id=None):

    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    model = HuggingFaceEmbeddings(model_name=model_name)

    persist_directory = f"./Chroma/{docstore_id}"
    vectorstore = Chroma.from_documents(
        documents=document_splits,
        embedding=model,
        persist_directory=persist_directory,
        collection_name=docstore_id,
    )


def get_vectordb(assistant_id):
    model_name = "sentence-transformers/all-MiniLM-L12-v2"
    model = HuggingFaceEmbeddings(model_name=model_name)

    vectordb = Chroma(
        persist_directory=f"./Chroma/{assistant_id}",
        collection_name=assistant_id,
        embedding_function=model,
    )
    return vectordb


def delete_docs_in_vectorbase(filename, vectorstore):
    filename = os.path.splitext(filename)[0]
    del_doc_ids = vectorstore.get(where={"filename": filename})["ids"]
    vectorstore.delete(ids=del_doc_ids)


# webpage title setting
st.set_page_config(page_title="OpenGPT Prototype", page_icon="📎")
st.subheader("Document Storage")

check_and_create(directory="./opengpt_app_data", filename="document_storage.csv")

ass_df = pd.read_csv("./opengpt_app_data/assistant_admin.csv")

ass_actions = st.selectbox(
    label="Assistant Options", options=ass_df["assistant_name"], index=None
)
if not ass_actions == None:
    doc_df = pd.read_csv("./opengpt_app_data/document_storage.csv")
    docs_store = doc_df[doc_df["assistant_name"] == ass_actions]["document"].values
    st.sidebar.write("Assistant document storage", docs_store)

    assistant_id = ass_df[ass_df["assistant_name"] == ass_actions][
        "assistant_id"
    ].values[0]

    st.info(f"Assistant Name : {ass_actions} , Assistant ID : {assistant_id}")

    actions = st.selectbox(label="Actions", options=["Add", "Delete"], index=None)

    # 助理檔案創建或刪除
    if actions == "Add":

        with st.form(key="form_fileloader", clear_on_submit=True):
            files = st.file_uploader(
                label="File Loader",
                type=["docx", "pptx", "csv", "pdf", "xlsx"],
                accept_multiple_files=True,
                label_visibility="collapsed",
                key="file_uploader",
            )
            form_btn_upload = st.form_submit_button("Upload", type="primary")

        if len(files) > 0 and form_btn_upload:
            docs = file_load(files)
            splits = file_splitter(docs)
            embedding_to_vector(splits, assistant_id)

            add_files = []
            for file in files:
                add_files.append(file.name)

            new_data = {
                "assistant_name": [ass_actions] * len(add_files),
                "document": add_files,
            }

            new_df = pd.DataFrame(new_data)

            doc_df = pd.concat([doc_df, new_df], ignore_index=True)

            st.write("Document has been uploaded.")

            doc_df.to_csv("./opengpt_app_data/document_storage.csv", index=False)

    elif actions == "Delete":

        vectordb = get_vectordb(assistant_id)

        doc_actions = st.selectbox(
            label="document Options",
            options=doc_df[doc_df["assistant_name"] == ass_actions]["document"],
            index=None,
        )

        if st.button("Delete"):
            doc_df = doc_df[
                (doc_df["assistant_name"] == ass_actions)
                & (doc_df["document"] != doc_actions)
            ]
            doc_df.to_csv(
                "./opengpt_app_data/document_storage.csv",
                index=False,
                encoding="utf-8-sig",
            )

            delete_docs_in_vectorbase(doc_actions, vectordb)

            st.write("Document has been removed.")

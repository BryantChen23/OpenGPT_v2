import streamlit as st
import string
import random
import pandas as pd
import os

from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


def generate_random_code(length):
    characters = string.ascii_letters + string.digits
    filder_id = "".join(random.choice(characters) for _ in range(length))
    return filder_id


def check_and_create(directory, filename):
    if not os.path.exists(directory):
        os.makedirs(directory)

    files = os.listdir(directory)

    if not filename in files:
        filepath = os.path.join(directory, filename)
        df = pd.DataFrame({"assistant_name": [], "assistant_id": []})
        df.to_csv(filepath, index=False, encoding="utf-8-sig")


# webpage title setting
st.set_page_config(page_title="OpenGPT Prototype", page_icon="📎")
st.subheader("Assistant Administrator")

check_and_create(directory="./opengpt_app_data", filename="assistant_admin.csv")

actions = st.selectbox(label="Actions", options=["Create", "Delete"])

if actions == "Create":
    ass_name = st.text_input(label="Assistant Name")
    ass_name = ass_name.strip()

    # If the 'Save' button is clicked
    if st.button("Save"):
        ass_df = pd.read_csv("./opengpt_app_data/assistant_admin.csv")
        if not ass_name:
            st.error(
                "Please provide a name, if you want to create a document assistant."
            )
        elif (ass_df["assistant_name"] == ass_name).any():
            st.error("oops, assistant name already exists.")
        else:
            # add assistant name and id data
            ass_id = generate_random_code(16)
            new_ass_data = pd.DataFrame(
                [{"assistant_name": ass_name, "assistant_id": ass_id}]
            )
            ass_df = pd.concat([ass_df, new_ass_data])
            ass_df.to_csv(
                "./opengpt_app_data/assistant_admin.csv",
                index=False,
                encoding="utf-8-sig",
            )

            # create a vector database
            persist_directory = f"./Chroma/{ass_id}"
            model_name = "sentence-transformers/all-MiniLM-L12-v2"
            model = HuggingFaceEmbeddings(model_name=model_name)
            vector = Chroma(
                persist_directory=persist_directory,
                collection_name=ass_id,
                embedding_function=model,
            )

            st.write("Your Assistant is ready.")
            st.write("Assistant Name : ", ass_name)
            st.write("Assistant Storage ID: ", ass_id)

elif actions == "Delete":
    ass_df = pd.read_csv("./opengpt_app_data/assistant_admin.csv")
    doc_df = pd.read_csv("./opengpt_app_data/document_storage.csv")
    temp_df = pd.read_csv("./opengpt_app_data/template_storage.csv")

    actions = st.selectbox(
        label="Assistant Options", options=ass_df["assistant_name"], index=None
    )
    if st.button("Delete"):
        ass_df = ass_df[ass_df["assistant_name"] != actions]
        ass_df.to_csv(
            "./opengpt_app_data/assistant_admin.csv", index=False, encoding="utf-8-sig"
        )

        doc_df = doc_df[doc_df["assistant_name"] != actions]
        doc_df.to_csv(
            "./opengpt_app_data/document_storage.csv", index=False, encoding="utf-8-sig"
        )

        temp_df = temp_df[temp_df["assistant_name"] != actions]
        temp_df.to_csv(
            "./opengpt_app_data/template_storage.csv", index=False, encoding="utf-8-sig"
        )

        st.write("Assistant has been removed.")

import json
import os
import time
from collections import OrderedDict

import openai
import pandas as pd
import streamlit as st
from langchain import HuggingFaceHub
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import MarkdownTextSplitter
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.vectorstores import Chroma
from modules.utils import add_bg_from_local, local_css, set_page_config


os.environ["GOOGLE_CSE_ID"] = st.secrets["GOOGLE_CSE_ID"]
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


if "messages" not in st.session_state:
    st.session_state.messages = OrderedDict()


def is_api_key_valid(model_host: str, api_key: str) -> bool:
    """
    Check if the provided API key is valid for the specified model host.

    Parameters:
        model_host (str): The name of the model host. Possible values are "openai" or "huggingface".
        api_key (str): The API key to be validated.

    Returns:
        bool: True if the API key is valid for the specified model host; False otherwise.
    """
    if api_key is None:
        st.sidebar.warning("Lütfen geçerli bir API keyi girin!", icon="⚠")
        return False
    elif model_host == "openai" and not api_key.startswith("sk-"):
        st.sidebar.warning("Lütfen geçerli bir OpenAI API keyi girin!", icon="⚠")
        return False
    elif model_host == "huggingface" and not api_key.startswith("hf_"):
        st.sidebar.warning("Lütfen geçerli bir HuggingFace API keyi girin!", icon="⚠")
        return False
    else:
        if model_host == "openai":
            os.environ["OPENAI_API_KEY"] = api_key
            openai.api_key = api_key
        else:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_key
        return True


def main():
    set_page_config()

    background_img_path = os.path.join("static", "background", "Sky BG.png")
    sidebar_background_img_path = os.path.join(
        "static", "background", "Lila Gradient.png"
    )
    page_markdown = add_bg_from_local(
        background_img_path=background_img_path,
        sidebar_background_img_path=sidebar_background_img_path,
    )
    st.markdown(page_markdown, unsafe_allow_html=True)

    css_file = os.path.join("static", "style.css")
    local_css(css_file)

    st.markdown(
        """<h1 style='text-align: center; color: black; font-size: 60px;'> 🤖 News Craft </h1>
        <br>
        <h3 style='text-align: center; color: black; font-size: 60px;'> Automated AI News Generator </h3>
        <br>""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        "<center><h1>Sohbet Botu Ayarları</h1></center> <br>",
        unsafe_allow_html=True,
    )
    llm_models = [
        "openai/gpt-3.5-turbo",
        "meta-llama/Llama-2-70b-chat-hf",
        "upstage/Llama-2-70b-instruct-v2",
        "upstage/Llama-2-70b-instruct",
        "stabilityai/StableBeluga2",
        "augtoma/qCammel-70-x",
        "google/flan-t5-xxl",
        "google/flan-ul2",
        "databricks/dolly-v2-3b",
        "Writer/camel-5b-hf",
        "Salesforce/xgen-7b-8k-base",
        "tiiuae/falcon-40b",
        "bigscience/bloom",
    ]
    model = st.sidebar.selectbox("Lütfen bir LLM seçin", llm_models)
    if model == "<Seçiniz>":
        st.sidebar.warning("Lütfen bir model seçin.")
        _, center_war_col, _ = st.columns([2, 5, 1])
        center_war_col.warning(
            "Lütfen sol taraftaki panelden bot için gerekli ayarlamaları yapın."
        )
        return
    else:
        model_host = "openai" if model.startswith("openai") else "huggingface"
        api_key = st.sidebar.text_input(
            f"Lütfen {model_host.title()} API keyini girin",
        )
        if is_api_key_valid(model_host, api_key):
            st.sidebar.success("API keyi başarıyla alındı.")
        else:
            _, center_war_col, _ = st.columns([2, 5, 1])
            center_war_col.warning(
                "Lütfen sol taraftaki panelden bot için gerekli ayarlamaları yapın."
            )
            return


if __name__ == "__main__":
    main()

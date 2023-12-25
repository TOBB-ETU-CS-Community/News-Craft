import base64

import streamlit as st

# imports for webscraper
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
import pandas as pd
import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta
from googleapiclient.discovery import build

# webscraper code

# google keys
my_api_key: str = "GOOGLE API KEY"
my_cse_id: str = "SEARCH ENGINE ID"

# openai key
# load_dotenv()
# llm=ChatOpenAI(temperature=0)

# region helpfunctions


def __google_search(
    search_term: str, api_key: str, cse_id: str, **kwargs
):  # gets the links
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()

    links = []
    for i in res["items"]:
        links.append(i["link"])

    return links


def __webPull(link: list):  # pulls data fram website
    loader = AsyncChromiumLoader(link)
    docs = loader.load()

    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs, tags_to_extract=["h2", "h3", "div", "span", "time"]
    )
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)

    return splits


# gives data to llm
def __extract(content: str, schema: dict, llm: ChatOpenAI):
    return create_extraction_chain(schema=schema, llm=llm).run(content)


def __parseDate(date_string: str):  # pareses the dates to the correct form
    try:
        current_datetime = datetime.datetime.now()

        if "ago" in date_string:
            time_difference = int(date_string.split()[0])
            parsed_date = current_datetime - relativedelta(days=time_difference)
        else:
            parsed_date = parser.parse(date_string, default=current_datetime)

        return parsed_date
    except:
        raise TypeError


def __list_to_df(
    input: list, site: str, reqDate: datetime.datetime, num
):  # creates the dataframe
    names = []
    dates = []
    siteN = []
    for obj in input:
        if obj["article_date"] != None:
            date = __parseDate(obj["article_date"])
            if date is not None:
                date = date.replace(tzinfo=None)
                reqDate = reqDate.replace(tzinfo=None)

                if date >= reqDate:
                    if len(names) >= num:
                        break

                    names.append(obj["article_title"])
                    dates.append(obj["article_date"])
                    siteN.append(site)

    df = pd.DataFrame({"Names": names, "Dates": dates, "Site": siteN})
    return df


# endregion helpfunctions


def scrape_news(time_range: str, selected_topics: str, num_articles: int):
    if type(selected_topics) != str:
        raise TypeError
    if num_articles < 0:
        raise IndexError
    if selected_topics is None or len(selected_topics) == 0:
        raise TypeError

    res = __google_search(selected_topics, my_api_key, my_cse_id)

    schema = {
        "properties": {
            "article_title": {"type": "string"},
            "article_date": {"type": "string"},
        },
        "required": ["article_title", "article_date"],
    }
    endDf = pd.DataFrame()
    index: int = 0
    lastDate = __parseDate(time_range)

    while index < len(res):
        links = [res[index]]

        linkSplit = res[index].split(".")
        linkName = linkSplit[1]
        splits = __webPull(link=links)

        extraction = __extract(splits[0], schema=schema, llm=llm)  # type list

        df = __list_to_df(extraction, linkName, reqDate=lastDate, num=num_articles)
        endDf = pd.concat([endDf, df])

        if len(endDf) >= num_articles:
            break

        index += 1

    if len(endDf) < num_articles:
        print("Not enough articles in the timespan!")

    return endDf


# END OF WEBSCRAPER


@st.cache_data
def add_bg_from_local(background_img_path, sidebar_background_img_path):
    with open(background_img_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    with open(sidebar_background_img_path, "rb") as image_file:
        sidebar_encoded_string = base64.b64encode(image_file.read())

    return f"""<style>
        .stApp {{
            background-image: url(data:image/png;base64,{encoded_string.decode()});
            background-size: cover;
        }}

        section[data-testid="stSidebar"] {{
            background-image: url(data:image/png;base64,{sidebar_encoded_string.decode()});
            background-size: cover;
        }}
        div[class="stChatFloatingInputContainer css-90vs21 ehod42b2"]
            {{
                background: url(data:image/png;base64,{encoded_string.decode()});
                background-size: cover;
                z-index: 1;
            }}
    </style>"""


def set_page_config():
    st.set_page_config(
        page_title="TOBB GPT",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/olympian-21",
            "Report a bug": "https://github.com/olympian-21",
            "About": """It is a chatbot powered by OpenAI, Langchain, ChromeDB, and
 Google APIs to educate students about TOBB University of Economics and Technology.""",
        },
    )


def local_css(file_name):
    # with open(file_name) as f:
    #    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    style = """<style>
        .row-widget.stButton {
            text-align: center;
            position: fixed;
            bottom: 0;
            z-index: 2;
            }
    </style>"""
    st.markdown(style, unsafe_allow_html=True)

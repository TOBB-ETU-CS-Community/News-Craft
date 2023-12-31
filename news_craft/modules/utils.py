import base64

import streamlit as st

# imports for webscraper
from dotenv import load_dotenv
import pandas as pd

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain

from datetime import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta
from langchain.schema.document import Document

#openai key
load_dotenv()
llm=ChatOpenAI(temperature=0)

#region helpfunctions
def __webPull(link:list,tags:list, split:bool=True, unwanted:list=[]): #pulls data fram website
    loader=AsyncChromiumLoader(link)
    docs=loader.load()

    bs_transformer=BeautifulSoupTransformer()
    docs_transformed=bs_transformer.transform_documents(
        docs,tags_to_extract=tags,unwanted_tags=unwanted
    )
    if split:
        splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=0)
        splits= splitter.split_documents(docs_transformed)
        return splits
    else:
        return docs_transformed
    
#gives data to llm
def __extract(content:str,schema:dict,llm:ChatOpenAI):
    return create_extraction_chain(schema=schema,llm=llm).run(content)

def __parseDate(date_string:str): #pareses the dates to the correct form
    try:
        current_datetime = datetime.now()
        tempList=date_string.split(" ")
        if(tempList[0]=="Published"):
            sum=""
            for i in range(1,len(tempList)):
                sum+=tempList[i]+" "
            
            date_string=sum[:-1]

        if 'ago' in date_string:

            time_difference = int(date_string.split()[0])
            parsed_date = current_datetime - relativedelta(days=time_difference)         
        else:
            parsed_date = parser.parse(date_string, default=current_datetime)

        return parsed_date
    except:
        return None


def __list_to_df(input:dict, sites : dict, reqDate:datetime, num): #creates the dataframe
    names=[]
    dates=[]
    siteN=[]
    for obj in input:
        
        if (obj["article_date"]!=None):
            date=__parseDate(obj["article_date"])
            if date is not None:  
                date = date.replace(tzinfo=None)
                reqDate = reqDate.replace(tzinfo=None)

                if (date >= reqDate): 
                    if (len(names)>=num):
                        break
                    if (obj["article_title"] in sites):
                        names.append(obj["article_title"])
                        dates.append(obj["article_date"])  
                        siteN.append(sites[obj["article_title"]])
    df=pd.DataFrame({
        "Names":names,
        "Dates":dates,
        "Site":siteN
    })
    return df


def inputSplitter(input:str): #sometimes scraped data comes without spaces and this function adds spaces accordingly
    if(input == None):
        return None
    i:int=0
    prevPointer : int=0
    titlesString=""
    while(i<len(input)-1):
        c:chr=input[i]
        afc: chr=input[i+1]
        if((c>='A' and c<='Z' and afc>='a' and afc <='z')):
            temp=input[prevPointer:i]
            spaceString=" "

            if(len(temp)<1):
                i+=1
                continue

            if(temp[-1]==" "):
                spaceString=""
            if(temp[:9]=="Published"):
                titlesString+=temp[:9] +" "+ temp[9:]+spaceString
            else:
                titlesString+=temp + spaceString
            
            prevPointer=i
        i+=1
    return titlesString

def linkSplitter(links:str):
    linksDic={}
    links = links.split(")")

    for linkAdress in links:
        #print(links)
        temp=linkAdress
        linkAdress=temp[1:]
        temp=linkAdress.split("(")
        if(len(temp[0])>2 and len(temp)>1):
            linksDic[temp[0][:-1]]=temp[1]
                
    return linksDic

def contentsPuller(df:pd.DataFrame): #gets the article content
    siteLinks:list=df.iloc[:,2].to_list()

    pull=__webPull(link=siteLinks,tags=["div","p"], split=False, unwanted=["h3","span","a","img","li","ul"])
    content = []
    for c in pull:
        content.append(inputSplitter(c.page_content))
        #sometimes script can't acces the website so if that happens we need to try again
        #when that error happens BeautifulSoup will give an error but there should be no problem in the output
        if(len(c.page_content)<2):
            return contentsPuller(df)

    df:pd.Series=pd.Series(content)
    df.name="Contents"
    return df

#endregion helpfunctions 

def scrape_news(time_range : str, selected_topics : str, num_articles : int):
    
    lastDate = __parseDate(time_range)
    
    if type(selected_topics) != str:
        raise TypeError
    if num_articles<=0:
        raise IndexError
    if selected_topics == None or len(selected_topics) == 0:
        raise TypeError
    if(lastDate== None):
        raise TypeError
    
    endDf:pd.DataFrame=pd.DataFrame()
    i: int = 1
    schema={
        "properties":{
            "article_title":{"type":"string"},
            "article_date":{"type":"string"},
        },
        "required":["article_title","article_date"]
    }
    while(len(endDf)<num_articles and i<11):
        bbc="https://www.bbc.co.uk/search?q="+selected_topics+"&d=NEWS_GNL&seqId=6dacb860-94ed-11ee-bf63-d95cb16fc5af&page="+str(i)

        links = [bbc]
        splits = __webPull(link=links,tags=["h3","div","span"],split=True)
        linksPuller : Document=__webPull(link=links,tags=["a"],split=True)

        linksDic: dict = linkSplitter(linksPuller[0].page_content[1091:])
        #print(linksDic)
        content = inputSplitter(splits[0].page_content)
        #print(content)
        extraction=__extract(content, schema=schema, llm=llm) #type list
        #print(extraction)
        df = __list_to_df(input=extraction, sites=linksDic, reqDate=lastDate, num=num_articles)
        endDf = pd.concat([endDf,df])

        i+=1
    
    if len(endDf) < num_articles:
        print("Not enough articles in the timespan!")


    articleContent=contentsPuller(endDf)
    endDf=pd.merge(df,articleContent,right_index=True,left_index=True)
    
    print(endDf)
    return endDf

# END OF WEBSCRAPER
# START OF SUMMARY FUNCTION
from enum import Enum
import pandas as pd

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage


load_dotenv()

class ComplexityLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"

class ReadingTime(Enum):
    ONE_MINUTE = "1 minute"
    TWO_MINUTES = "2 minutes"
    FIVE_MINUTES = "5 minutes"

#promts
templateStory="""If the given information is not enough, say "I don't know".
You are an editor for the news articles and you give summaries accordingly to the given template. With an academic tone and language. 
Here is the construction of your summary:

Title

Abstract: Here you describe the article with one sentence.
Key Words: Some keywords about the context of the article.
Main Summary: Deep dive into the article make a complete summary here.
Results: In this section explain what might be the consequences of this event.
Conclusion: Make a summary of the main points of the article.
Extra: Evaluate the relevance of the news to the audience or community.

Here is the input to give output from:
Article: {article}
Summary:
"""
templateBulletPoint="""If the given information is not enough, say "I don't know".
You are an editor for the news articles and you give summaries based on the template given below.

These are some bullet points you should use in the summary:

-Identify the primary facts - who, what, when, why, and how.
-Look for background information that provides a broader understanding of the topic or event.
-Explore the potential implications or consequences of the event or topic discussed.
"""
extraBulletPoints="""-Analyze the tone of the article and consider any potential biases in the reporting.
-Evaluate the relevance of the news to the audience or community.
-Assess the credibility and diversity of sources used in the article"""
bulletPointArticlePart="""

Here is the input to give output from:
Article: {article}
Summary:"""


def generate_summaries(news_dataframe: pd.DataFrame, complexity_level: ComplexityLevel, reading_time: ReadingTime) -> pd.DataFrame:
    if news_dataframe.empty:
        raise ValueError
    if (news_dataframe.columns.size<4):
        raise ValueError
    
    #decides the output style
    temperature : float=0
    if(complexity_level==ComplexityLevel.HARD):
        temperature=0.3
        template=templateStory
    else:
        if(ReadingTime.ONE_MINUTE==reading_time):
            template=templateBulletPoint+bulletPointArticlePart
        else:
            template=templateBulletPoint+extraBulletPoints+bulletPointArticlePart
            if(reading_time==ReadingTime.TWO_MINUTES):
                temperature=0.3

    prompt_template = PromptTemplate(input_variables=["article"], template=template)

    llm : ChatOpenAI=ChatOpenAI(temperature=temperature , model="gpt-3.5-turbo-1106")

    #MEMO openai api does not allow more than 3 requests in one minute on free accounts
    output=[]

    #there is a token limit in llm so this loop ensures input stays between the limit by just sending one at a time
    for i in range(0,len(news_dataframe)):
        #when working on csv file it gets longer by one column so change this index
        doc = news_dataframe.iloc[i,3]
        messages=[HumanMessage(content=prompt_template.format(article=doc))]
        outputMessage=llm.invoke(messages).content
        
        #if content is empty llm will answer "I don't know." this ensures its warning
        if(outputMessage=="I don't know."):
            raise ValueError("Content of the article is not compatible for summary")
        
        output.append(outputMessage)
    
    temp:pd.DataFrame=news_dataframe.iloc[:,0:-1]
    outputToSeries=pd.Series(output)
    outputToSeries.name="Summaries"
    df=pd.merge(temp,outputToSeries,right_index=True,left_index=True)


    #print(df)
    return df

#END OF SUMMARY FUNCTION

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

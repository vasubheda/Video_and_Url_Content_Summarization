import validators,streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader,UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Streamlit app
st.set_page_config(page_title="Langchain: Summarize Text From YT or website",page_icon="🦜")
st.title("🦜 Langchain: Summarize Text From YT or Website")
st.subheader("Summarizer URL")

## Get the groq api key and the url(YT or Website) to be summarized
with st.sidebar:
    groq_api_key=st.text_input("Groq Api Key",value="",type="password")
    
generic_url=st.text_input("URL",label_visibility="collapsed")

prompt_template="""
Provide a summary of the following content in 300 words:
Content:{text}
"""
prompt=PromptTemplate(template=prompt_template,input_variables=["text"])

final_prompt="""
Provide the final summary of all the content in 300 words with these important points.
Add a Motivation title, Start the precise summary with an introduction and then provide 
the summary of the content in a structured way.
Content:{text}
"""

final_prompt_template=PromptTemplate(input_variables=['text'],template=final_prompt)

if st.button("Summarize the Content from YT or Website"):
    ## Validate all the inputs
    if not groq_api_key.strip() or not generic_url.strip():
        st.error("Please provide the information to get started")
        st.stop()
    elif not validators.url(generic_url):
        st.error("Please enter a valid Url. It can be a YT video url or website url")
    else:
        try:
            with st.spinner("Waiting...."):
                #loading the website or yt video data
                ## Gemma model using Groq Api
                llm=ChatGroq(model="Gemma-7b-It",groq_api_key=groq_api_key)
                if "youtube.com" in generic_url:
                    loader=YoutubeLoader.from_youtube_url(generic_url,add_video_info=True)
                else:
                    loader=UnstructuredURLLoader(urls=[generic_url],ssl_verify=False,
                                                 headers={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs=loader.load()
                
                final_docs=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=100).split_documents(docs)
                
                ## Chain for summarization
                chain=load_summarize_chain(llm,
                                           chain_type="map_reduce",
                                           map_prompt=prompt,
                                           combine_prompt=final_prompt_template
                                           )
                output_summary=chain.run(final_docs)
                
                st.success(output_summary)
        except Exception as e:
            st.exception(f"Exception:{e}")        
        
import os, tempfile
from pathlib import Path

from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory

import streamlit as st

# 定義目錄路徑
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.joinpath('data')
TMP_DIR = DATA_DIR.joinpath('tmp')
LOCAL_VECTOR_STORE_DIR = DATA_DIR.joinpath('vector_store')

# 創建必要的目錄
def ensure_directories():
    """確保所有必要的目錄都存在"""
    for dir_path in [DATA_DIR, TMP_DIR, LOCAL_VECTOR_STORE_DIR]:
        dir_path.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="RAG")
st.title("Streamlit Showcase: Unleashing the Power of RAG and LangChain")
mode = st.sidebar.radio(
    "LLM type：",
    ('Your own LLM', 'openAI'))
if mode == 'Your own LLM':
    openai_api_base = st.sidebar.text_input('URL:', type='default')
    openai_api_key = 'None'
elif mode == 'openAI':
    openai_api_base = st.sidebar.text_input('api_base:', type='password')
    openai_api_key = st.sidebar.text_input('key:', type='password')

def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    model_name = r"E:\pythonwork\RAG_LangChain_streamlit\all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name,
                                    model_kwargs=model_kwargs)
    vectordb = Chroma.from_documents(texts, embedding=embeddings,
                                     persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def define_llm():
    llm = ChatOpenAI(openai_api_key=openai_api_key, openai_api_base=openai_api_base)
    return llm

def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm = define_llm(),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result

def query_llm_direct(query):
    llm = define_llm()
    llm_chain = add_prompt(llm, query)
    result = llm_chain.invoke({"query": query})
    result = result['text']
    st.session_state.messages.append((query, result))
    return result

def add_prompt(llm, query):
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    init_Prompt = """
    you are helpful, kind, honest, good at writing, and never fails to answer any requests immediately and with precision. \
    Provide an answer to the following question in about 150 words. Ensure that the answer is informative, \
    relevant, and concise: \
    {query}
    """
    
    input_prompt = PromptTemplate(input_variables=["query"], template=init_Prompt)
    return LLMChain(prompt=input_prompt, llm=llm)

def input_fields():
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)

def process_documents():
    if not openai_api_base or not openai_api_key:
        st.warning(f"Please provide information about LLM model.")
    else:
        try:
            for source_doc in st.session_state.source_docs:
                # 確保臨時文件被保存在 tmp 目錄中
                with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                    tmp_file.write(source_doc.read())
                
                documents = load_documents()
                
                # 清理臨時文件
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                
                texts = split_documents(documents)
                st.session_state.retriever = embeddings_on_local_vectordb(texts)
        except Exception as e:
            st.error(f"An error occurred: {e}")

def boot():
    # 確保必要的目錄存在
    ensure_directories()
    
    # 顯示輸入欄位
    input_fields()
    
    # 添加提交按鈕
    st.button("Submit Documents", on_click=process_documents)
    
    # 初始化消息歷史
    if "messages" not in st.session_state:
        st.session_state.messages = []    
    
    # 顯示歷史消息
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])    
    
    # 處理新的查詢
    if query := st.chat_input():
        st.chat_message("human").write(query)

        if "retriever" in st.session_state:
            response = query_llm(st.session_state.retriever, query)
        else:
            response = query_llm_direct(query)

        st.chat_message("ai").write(response)

if __name__ == '__main__':
    boot()
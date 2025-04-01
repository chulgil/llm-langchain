import streamlit as st
import os

from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_upstage import UpstageEmbeddings
from langchain.chains import RetrievalQA
from dotenv import load_dotenv


st.set_page_config(page_title="소득세 챗봇", page_icon=":shark:")
st.title("소득세 챗봇")
st.caption("소득세 관련 질문에 대해 답해드립니다.")

def get_ai_message(user_message) :
    
    load_dotenv()
    # OpenAI에서 제공하는 Embedding Model을 활용해서 `chunk`를 `vector`로 변환
    embedding = UpstageEmbeddings(model='solar-embedding-1-large')

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, # 문서를 쪼갤때 하나의 하나의 청크가 가질수 있는 토큰수
        chunk_overlap=200, # 문서를 가져올때 1부터 10라인까지 가져오면 그후 2부터 11 라인을 가져오는 식으로 중복되는 라인수
    )

    loader = Docx2txtLoader('../data/tax.docx')
    document_list = loader.load_and_split(text_splitter=text_splitter)
    len(document_list)

    index_name = 'tax-upstage-table-index'
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    # pc = Pinecone(api_key=pinecone_api_key)
    database = PineconeVectorStore.from_documents(document_list, embedding, index_name=index_name)

    llm = ChatOpenAI(model='gpt-4o')
    prompt = hub.pull("rlm/rag-prompt")
    retriever = database.as_retriever(search_kwargs={'k': 4})
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    
    dictionary = ["사람을 나타내는 표현 -> 거주자"]
    prompt = ChatPromptTemplate.from_template(f"""
        사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요.
        만약 변경할 필요가 없다고 판단된다면, 사용자의 질문을 변경하지 않아도 됩니다.
        그런 경우에는 질문만 리턴해주세요.
        사전: {dictionary}
        
        질문: {{question}}
    """)
    dictionary_chain = prompt | llm | StrOutputParser()
    tax_chain = {"query": dictionary_chain} | qa_chain
    
    ai_message = tax_chain.invoke({"question": user_message})
    return ai_message["result"]

if 'messages' not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        

    
print(f"before: {st.session_state.messages}")


if user_question := st.chat_input(placeholder="질문을 입력하세요"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})    
    
    with st.spinner("답변을 생성하는 중입니다."):
        ai_message = get_ai_message(user_question)
        with st.chat_message("assistant"):
            st.write(ai_message)
        st.session_state.messages.append({"role": "assistant", "content": ai_message})


print(f"after: {st.session_state.messages}")


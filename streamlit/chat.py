import streamlit as st
import os

from langchain_community.vectorstores import Pinecone
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

from llm import get_ai_message


st.set_page_config(page_title="소득세 챗봇", page_icon=":shark:")
st.title("소득세 챗봇")
st.caption("소득세 관련 질문에 대해 답해드립니다.")

load_dotenv()

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


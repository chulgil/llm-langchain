import streamlit as st

st.set_page_config(page_title="소득세 챗봇", page_icon=":shark:")
st.title("소득세 챗봇")
st.caption("소득세 관련 질문에 대해 답해드립니다.")

if 'messages' not in st.session_state:
    st.session_state.messages = []
    
print(f"before: {st.session_state.messages}")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        

if user_question := st.chat_input(placeholder="질문을 입력하세요"):
    with st.chat_message("user"):
        st.write(user_question)
    st.session_state.messages.append({"role": "user", "content": user_question})    
    
    with st.chat_message("assistant"):
        st.write("AI Message")
    st.session_state.messages.append({"role": "assistant", "content": "답변을 입력하고 있습니다..."})


print(f"after: {st.session_state.messages}")


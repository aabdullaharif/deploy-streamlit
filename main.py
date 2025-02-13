import os
import streamlit as st

import pysqlite3
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

def main():
    st.set_page_config(page_title="Let's talk about JRB!", page_icon="🤖", layout="centered")
    st.title("Let's talk about JRB!")

    # API Key Input
    api_key = st.text_input("Enter your OpenAI API key:", type="password")

    if api_key:
        # Initialize embeddings and chat model with the API key
        embeddings = OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small")
        db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

        retriever = db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 10, "fetch_k": 50, "threshold": 0.5}
        )

        llm = ChatOpenAI(model="gpt-4o", api_key=api_key)

        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just "
            "reformulate it if needed and otherwise return it as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_system_prompt = (
            "You are an AI assistant designed for question-answering tasks "
            "using Slack channel conversations as context. Your goal is to "
            "understand and extract relevant answers from the retrieved chat history. "
            "\n\n"
            "The conversation format follows this pattern:\n"
            "[Sender Name] (YYYY-MM-DD): Message\n\n"
            "Guidelines:\n"
            "- Provide direct answers based on the retrieved context.\n"
            "- If a message is a response to a previous one, interpret it in that context.\n"
            "- Use your mind to understand the context and provide accurate answers.\n"
            "- If you need more context, ask for it.\n"
            "- Give a response that is relevant to the context.\n"
            "Context:\n{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create the QA and RAG chains
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        user_input = st.chat_input("Type your message here...")
        if user_input:
            st.session_state["chat_history"].append({"role": "user", "content": user_input})

            with st.chat_message("user"):
                st.markdown(user_input)

            chain_history = []
            print(st.session_state["chat_history"])
            for msg in st.session_state["chat_history"]:
                if msg["role"] == "user":
                    chain_history.append(HumanMessage(content=msg["content"]))
                else:
                    chain_history.append(SystemMessage(content=msg["content"]))

            with st.spinner("AI is typing..."):
                result = rag_chain.invoke({"input": user_input, "chat_history": chain_history})
                answer = result["answer"]

            st.session_state["chat_history"].append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

if __name__ == "__main__":
    main()

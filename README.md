
# GenAI

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import AzureOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# --- Setup Azure OpenAI ---
os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]
os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"]
os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = st.secrets["AZURE_OPENAI_DEPLOYMENT_NAME"]
os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"] = st.secrets["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"]

llm = AzureOpenAI(
    deployment_name=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    model_name="gpt-35-turbo",  # Or the model you are using
    temperature=0,
)

embeddings = AzureOpenAIEmbeddings(
    deployment_name=os.environ["AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME"],
    model="text-embedding-ada-002",  # Or the embedding model you are using
)

# --- Function to load and process PDF documents ---
def load_and_process_pdfs(pdf_files):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    all_texts = []
    for pdf_file in pdf_files:
        loader = PyPDFLoader(pdf_file)
        data = loader.load()
        texts = text_splitter.split_documents(data)
        all_texts.extend(texts)
    return all_texts

# --- Function to create the vector store ---
def create_vector_store(texts, embeddings):
    vector_store = Chroma.from_documents(texts, embeddings)
    return vector_store

# --- Initialize the conversation chain ---
def initialize_conversation_chain(vector_store, llm, memory):
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )
    return qa

# --- Main Streamlit application ---
def main():
    st.title("RAG-based Conversational Chatbot with Multiple PDFs")

    # Upload multiple PDF files
    pdf_files = st.file_uploader("Upload multiple PDF files", type=["pdf"], accept_multiple_files=True)

    if pdf_files:
        with st.spinner("Processing PDF files..."):
            texts = load_and_process_pdfs(pdf_files)
            vector_store = create_vector_store(texts, embeddings)
            st.session_state["vector_store"] = vector_store
            st.success("PDF files processed and indexed successfully!")

    if "vector_store" in st.session_state:
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you with the information in these PDFs?"}]
        if "chat_memory" not in st.session_state:
            st.session_state["chat_memory"] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
            st.session_state["qa"] = initialize_conversation_chain(st.session_state["vector_store"], llm, st.session_state["chat_memory"])

        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask me anything about the PDFs!"):
            st.session_state["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state["qa"]({"question": prompt})
                    response = result["answer"]
                    st.markdown(response)
                    st.session_state["messages"].append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

import streamlit as st
from PyPDF2 import PdfReader
from langchain.embeddings import OpenAIEmbeddings #HuggingFaceInstructEmbeddings
#from langchain.vectorstores import Chroma

from langchain.vectorstores import FAISS
from dotenv import load_dotenv , find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
from langchain.text_splitter import CharacterTextSplitter
#Add memory
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size = 500,
        chunk_overlap = 100,
        length_function = len
        )
    chunks = text_splitter.split_text(text)
    return chunks

def get_embedding(text_chunks):
     #persist_directory = 'db'
     embeddings = OpenAIEmbeddings()
     
     vectordb = FAISS.from_texts(texts = text_chunks,
                                  embedding=embeddings,
                                  )
     return vectordb

def get_conversation(vectordb):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(),
        memory=memory
    )
    return conversation_chain




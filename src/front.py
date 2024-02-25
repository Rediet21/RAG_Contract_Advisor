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
from template import css, bot_template, user_template
from utils.utils import get_pdf_text, get_text_chunks, get_embedding, get_conversation

# def get_pdf_text(pdf_docs):
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

# def get_text_chunks(text):
#     text_splitter = CharacterTextSplitter(
#         separator="\n",
#         chunk_size = 500,
#         chunk_overlap = 100,
#         length_function = len
#         )
#     chunks = text_splitter.split_text(text)
#     return chunks

# def get_embedding(text_chunks):
#      #persist_directory = 'db'
#      embeddings = OpenAIEmbeddings()
     
#      vectordb = FAISS.from_texts(texts = text_chunks,
#                                   embedding=embeddings,
#                                   )
#      return vectordb

# def get_conversation(vectordb):
#     llm = ChatOpenAI()
#     memory = ConversationBufferMemory(memory_key = 'chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectordb.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain


def handle_input(query, conversation):
    response = conversation({'question': query})
    st.session_state.chat_history = response['chat_history']
    st.write(response)

def main():
    st.set_page_config(page_title="RAG", page_icon=":pdf:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    st.header("Chat with the PDF")
    query = st.text_input("Ask a question about your documents:")
    if query:
        handle_input(query, st.session_state.conversation)
    st.write(user_template, unsafe_allow_html=True)
    st.write(bot_template, unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your document")
        pdf_docs = st.file_uploader("Upload your files", accept_multiple_files=True)
        st.file_uploader('Your pdf here')
        if st.button("Process"):
            st.spinner("Loading")
            raw_text = get_pdf_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            vectordb = get_embedding(text_chunks)
            st.session_state.conversation = get_conversation(vectordb)



if __name__ == '__main__':
    main()
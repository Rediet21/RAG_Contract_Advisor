from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from dotenv import load_dotenv , find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

file = PdfReader('./data/Robinson Advisory.pdf')
file

raw_text = ''

for i, page in enumerate (file.pages):
    text = page.extract_text()
    if text:
        raw_text += text

len (raw_text)
raw_text[:1000]

# Splitting up the text into smaller chunks for indexing
text_splitter = CharacterTextSplitter(        
    separator = ",",
    chunk_size = 500,
    chunk_overlap  = 0, #striding over the text
    length_function = len,
)
texts = text_splitter.create_documents(raw_text)



len(texts)

# Embedding using openAI

embedding = OpenAIEmbeddings()
#directory to store the embedding on disk
persist_directory = 'db'

vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embedding,
                                 persist_directory=persist_directory)

# persiste the db to disk
vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)
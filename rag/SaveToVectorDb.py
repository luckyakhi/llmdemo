# Split the document content into smaller chunks if needed
import langchain
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
import os

from rag.PdfLoader import get_pdf_contents
load_dotenv()
# Initialize embeddings
embeddings = OpenAIEmbeddings()

# Initialize Pinecone vector store
index_name = os.environ.get('PINECONE_INDEX')
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

text_splitter = CharacterTextSplitter(chunk_size=2048, chunk_overlap=0)
docs = text_splitter.split_text(get_pdf_contents())

# Add documents to Pinecone
vectorstore.add_texts(docs)

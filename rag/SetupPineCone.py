from dotenv import load_dotenv
import os

load_dotenv()


from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings

# Initialize embeddings

embeddings = OpenAIEmbeddings(dimensions=1536)

# Initialize Pinecone vector store
index_name = os.environ.get('PINECONE_INDEX')
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

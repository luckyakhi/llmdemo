import os
from typing import List

from chromadb.api.types import EmbeddingFunction, Documents
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

class CustomOpenAIEmbeddings(EmbeddingFunction[Documents]):
    def __init__(self, ef):
        self.ef = ef

    def __call__(self, input: Documents) -> list:
        return self.ef.embed_documents(input)

    def embed_query(self, text: str) -> List[float]:
        return self.ef.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.ef.embed_documents(texts)

# Initialize the embedding function
load_dotenv()
langchain_embeddings = OpenAIEmbeddings(model="text-embedding-ada-002",api_key=os.getenv("OPENAI_API_KEY"))
embedding_function = CustomOpenAIEmbeddings(langchain_embeddings)

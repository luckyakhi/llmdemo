import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI
import os

from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()
# Initialize Pinecone vector store
index_name = os.environ.get('PINECONE_INDEX')
# Initialize embeddings
embeddings = OpenAIEmbeddings()

vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
# Initialize LLM
llm = OpenAI()

# Initialize RAG chain
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Invoke the RAG chain with a question
query = ("Answer based on the attached PDF only, "
         "If the PDF doesnt contain the most likely response for query say I dont know. "
         "Can depersonalization disorder cause mental fatigue ")
answer = qa.invoke(query)

print(answer['result'])

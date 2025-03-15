import chromadb
from dotenv import load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAI

from rag.CustomEmbedding import embedding_function


def setup_rag_pipeline():
    # Connect to ChromaDB server
    client = chromadb.HttpClient(host="http://localhost:8000")

    # Create or get a collection
    collection_name = "research_papers"
    collection = client.get_or_create_collection(collection_name)

    # Create a Chroma vector store
    vector_store = Chroma(
        client=client,
        collection_name=collection_name,
        create_collection_if_not_exists=True
        ,embedding_function=embedding_function
    )

    # Create a vector store retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Initialize the language model
    llm = OpenAI(temperature=0)

    # Define a prompt template for RAG
    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    Question: {input}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Create the question-answering chain
    #combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    qa_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # Create the RAG chain
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=qa_chain)

    return rag_chain

def query_rag_pipeline(rag_chain, query):
    """Query the RAG pipeline with a user prompt."""
    result = rag_chain.invoke({"input": query})
    return result

if __name__ == '__main__':
    load_dotenv()
    rag_chain = setup_rag_pipeline()
    query = "What is the main topic of the document?"
    answer = query_rag_pipeline(rag_chain, query)
    print(answer)

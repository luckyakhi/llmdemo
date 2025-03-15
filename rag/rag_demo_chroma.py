import chromadb
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_openai import OpenAI


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
    )

    # Create a vector store retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Initialize the language model
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Define a prompt template for RAG
    template = """You are an assistant for question-answering tasks.
    Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Create the RAG chain
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        prompt=prompt,
        return_source_documents=True
    )

    return rag_chain

def query_rag_pipeline(rag_chain, query):
    """Query the RAG pipeline with a user prompt."""
    answer = rag_chain.run(query)
    return answer

if __name__ == '__main__':
    rag_chain = setup_rag_pipeline()
    query = "What is the main topic of the document?"
    answer = query_rag_pipeline(rag_chain, query)
    print(answer)

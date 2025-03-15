import pdfplumber
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAI, OpenAIEmbeddings

load_dotenv()

def extract_text_from_pdf(file_path):
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return ""

def create_in_memory_vector_store(text, chunk_size=512):
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    embedding_function = OpenAIEmbeddings()
    vector_store = InMemoryVectorStore(embedding_function)
    documents = [
        Document(id=str(i), page_content=chunk, metadata={})
        for i, chunk in enumerate(chunks)
    ]
    vector_store.add_documents(documents=documents)
    return vector_store

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

def create_rag_chain(vector_store):
    llm = OpenAI()
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="Answer the question: {question} based on the retrieved documents."
    )
    rag_chain = (
            {"context": retriever | format_docs, "question": lambda x: x}
            | prompt_template
            | llm
            | StrOutputParser()
    )
    return rag_chain

def main():
    pdf_file_path = "Files/DPTreatment.pdf"
    text = extract_text_from_pdf(pdf_file_path)
    if not text:
        print("No text extracted from PDF.")
        return

    vector_store = create_in_memory_vector_store(text)
    rag_chain = create_rag_chain(vector_store)

    def ask_question(question):
        return rag_chain.invoke({"question": question})

    answer = ask_question("What is the main topic of the document?")
    print(answer)

if __name__ == "__main__":
    main()

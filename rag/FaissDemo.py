import pdfplumber
import openai
import numpy as np
import faiss
from langchain.llms import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.runnables import RunnablePassthrough
from langchain.output_parsers import StrOutputParser


def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text


def create_openai_embeddings(text, chunk_size=512):
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    openai.api_key = "YOUR_OPENAI_API_KEY"
    embeddings = []
    for chunk in chunks:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=chunk
        )
        embeddings.append(response["data"][0]["embedding"])
    return chunks, embeddings


def create_faiss_index(embeddings):
    embeddings_array = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embeddings_array.shape[1])
    index.add(embeddings_array)
    return index


def create_rag_chain(chunks, index):
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)

    def retrieve(question):
        openai.api_key = "YOUR_OPENAI_API_KEY"
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=question
        )
        question_embedding = response["data"][0]["embedding"]
        question_embedding_array = np.array([question_embedding]).astype('float32')
        distances, indices = index.search(question_embedding_array, k=5)
        return [chunks[i] for i in indices[0]]

    prompt_template = PromptTemplate(
        input_variables=["question"],
        template="Answer the question: {question} based on the retrieved documents."
    )

    rag_chain = (
            {"context": retrieve, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
    )

    return rag_chain


def main():
    pdf_file_path = "path/to/your/document.pdf"
    text = extract_text_from_pdf(pdf_file_path)
    chunks, embeddings = create_openai_embeddings(text)
    index = create_faiss_index(embeddings)
    rag_chain = create_rag_chain(chunks, index)

    def ask_question(question):
        return rag_chain({"question": question})

    answer = ask_question("What is the main topic of the document?")
    print(answer)


if __name__ == "__main__":
    main()

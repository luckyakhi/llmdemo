import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import chromadb

def pdf_to_text(pdf_path):
    """Convert PDF to text."""
    pdf_file_obj = open(pdf_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file_obj)
    text = ''
    for page in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page].extract_text()
    pdf_file_obj.close()
    return text

def upload_pdf_to_chromadb(pdf_path):
    """Upload PDF to ChromaDB."""
    # Convert PDF to text
    text = pdf_to_text(pdf_path)

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = splitter.create_documents([text])

    # Generate embeddings
    embeddings = OpenAIEmbeddings()

    # Connect to ChromaDB server
    client = chromadb.HttpClient(host="http://localhost:8000")

    # Create or get a collection
    collection_name = "research_papers"
    collection = client.get_or_create_collection(collection_name)

    # Add documents to the collection
    for i, doc in enumerate(docs):
        # Generate embedding for the document
        embedding = embeddings.embed_documents([doc.page_content])[0]

        # Manually assign an ID to the document
        doc_id = f"pdf_{i}"

        # Add document to the collection
        collection.add(
            documents=[doc.page_content],
            metadatas=[{"source": "pdf"}],
            ids=[doc_id],
            embeddings=[embedding]
        )

if __name__ == '__main__':
    load_dotenv()
    pdf_path = 'files/DPTreatment.pdf'
    upload_pdf_to_chromadb(pdf_path)

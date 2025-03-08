from langchain_community.document_loaders import PyPDFLoader

# Load PDF document
loader = PyPDFLoader("files/DPTreatment.pdf")
documents = loader.load_and_split()

# Concatenate document content into a single string
def concatenate_docs(docs):
    content = "\n\n".join([doc.page_content for doc in docs])
    return content

document_content = concatenate_docs(docs=documents)
print(document_content)

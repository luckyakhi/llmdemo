from langchain_community.document_loaders import PyPDFLoader



# Concatenate document content into a single string
def concatenate_docs(docs):
    content = "\n\n".join([doc.page_content for doc in docs])
    return content

def get_pdf_contents():
    # Load PDF document
    loader = PyPDFLoader("files/DPTreatment.pdf")
    documents = loader.load_and_split()
    return concatenate_docs(docs=documents)

if __name__ == '__main__':
    print(get_pdf_contents())
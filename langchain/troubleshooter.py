import os

from dotenv import load_dotenv
from langchain_openai import AzureOpenAI

load_dotenv()
model = AzureOpenAI(azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"))
response = model.invoke("The sky is")
print(response.content)
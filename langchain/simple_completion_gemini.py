# Ensure your VertexAI credentials are configured
from dotenv import load_dotenv
from langchain_google_vertexai import ChatVertexAI

load_dotenv()
model =ChatVertexAI(model="gemini-2.0-flash-001",verbose=False)
response = model.invoke("The sky is")
print(response.content)
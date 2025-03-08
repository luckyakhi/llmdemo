import pinecone
from dotenv import load_dotenv
import os

load_dotenv()
# Initialize Pinecone client
#pinecone.init(api_key='YOUR_API_KEY', environment='YOUR_ENVIRONMENT')

# Delete the existing index
pinecone.Index(os.environ["PINECONE_INDEX"]).delete()

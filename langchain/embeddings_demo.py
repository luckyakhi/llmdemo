from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
open_ai_embeddings = OpenAIEmbeddings()
embeddings = open_ai_embeddings.embed_documents(["Cricket is a game of skill and mental fortitude",
                                                 "It has three different formats T20 , ODI and Tests",
                                                 "Wicket are made good for batting in the order of T20 ODI and Tests"])
print(embeddings)
#print embeddings dimension
print("Embedding Dimension: ", len(embeddings[0]))
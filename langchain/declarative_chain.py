from langchain_core.prompts import PromptTemplate

from langchain.model_utils import init_and_get_model

model = init_and_get_model()
prompt = PromptTemplate.from_template("""
Answer the user query based on the context given:
query: {query}
context: {context}
If information is not present in context , reply with I don't know 
""")
chain = prompt | model
result = chain.invoke({"query":"Who won the IPL 2025","context":"RCB won the IPL 2025"})
print(result.content)
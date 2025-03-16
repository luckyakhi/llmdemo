from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import chain
from langchain_openai import OpenAI

from langchain.model_utils import init_and_get_model

template_string = """
    Respond to the query: {query} by taking information only from context:{context}.
    If information is not present in context provide standard reply 'Information not present in context'
"""
model = init_and_get_model()
@chain
def call_in_chain(input):

    template = ChatPromptTemplate.from_template(template=template_string)
    prompt = template.invoke(input)
    return model.invoke(input=prompt)
if __name__ == '__main__':
    query = "What is the Capital of Hindustan"
    context = "As of recent update Capital of Hindustan is Bangalore"
    result = call_in_chain.invoke({"query":query,"context":context})
    print(result.content)


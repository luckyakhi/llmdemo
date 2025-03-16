from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

from langchain.model_utils import init_and_get_model


def simple_prompt_template():
    template = PromptTemplate.from_template("""Answer the question based on the 
    context below. If the question cannot be answered using the information 
    provided, answer with "I don't know".
Context: {context}
Question: {question}
Answer: """)
    model = init_and_get_model()
    prompt = template.invoke({
        "context": """The most recent advancements in NLP are being driven by Large
        Language Models (LLMs). These models outperform their smaller 
        counterparts and have become invaluable for developers who are creating 
        applications with NLP capabilities. Developers can tap into these 
        models through Hugging Face's `transformers` library, or by utilizing 
        OpenAI and Cohere's offerings through the `openai` and `cohere` 
        libraries, respectively.Recently X started providing LLMs through its Grok offering""",
        "question": "Which model providers offer LLMs?"
    })
    completion = model.invoke(prompt)
    print(completion.content)

def prompt_template_with_message_types():
    template = ChatPromptTemplate.from_messages([
        ('system', '''Answer the question based on the context below. If the 		
            question cannot be answered using the information provided, answer
            with "I don\'t know".'''),
        ('human', 'Context: {context}'),
        ('human', 'Question: {question}'),
    ])

    model = init_and_get_model()

    # `prompt` and `completion` are the results of using template and model once

    prompt = template.invoke({
        "context": """The most recent advancements in NLP are being driven by 
            Large Language Models (LLMs). These models outperform their smaller 
            counterparts and have become invaluable for developers who are creating 
            applications with NLP capabilities. Developers can tap into these 
            models through Hugging Face's `transformers` library, or by utilizing 
            OpenAI and Cohere's offerings through the `openai` and `cohere` 
            libraries, respectively.""",
        "question": "Which model providers offer LLMs?"
    })
    completion = model.invoke(prompt)
    print(completion.content)

if __name__ == '__main__':
    #simple_prompt_template()
    prompt_template_with_message_types()
    
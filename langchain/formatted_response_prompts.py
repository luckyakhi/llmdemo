from langchain.model_utils import init_and_get_model
from langchain.response_schema import AnswerWithReferences

if __name__ == '__main__':
    model = init_and_get_model(model_key = "OPENAI_STRUCTURED_MODEL")
    schema = AnswerWithReferences
    structured_llm = model.with_structured_output(schema=schema)
    output = structured_llm.invoke("Can weed trigger panic attack")
    print(output.answer)
    print(output.references)


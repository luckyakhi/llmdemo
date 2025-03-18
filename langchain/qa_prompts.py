from langchain_core.messages import SystemMessage, HumanMessage

from langchain.model_utils import init_and_get_model, get_azure_model


def simple_invoke():
    response = get_azure_model().invoke("The sky is")
    print(response.content)

def invoke_with_message_types():
    system_msg = SystemMessage(
        '''You are a helpful assistant that responds to questions with three 
            exclamation marks.'''
    )
    human_msg = HumanMessage('What is the capital of France?')
    response = init_and_get_model().invoke([system_msg, human_msg])
    print(response.content)

if __name__ == '__main__':
    invoke_with_message_types()
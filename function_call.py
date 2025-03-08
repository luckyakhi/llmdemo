from openai import OpenAI
# Parse the arguments into a Python dictionary
import json


def concat(str1,str2):
    return f"{str1}-{str2}"
functions = [
    {
        "name": "concat",
        "description": "A custom function that concats arguments and returns a result.",
        "parameters": {
            "type": "object",
            "properties": {
                "str1": {
                    "type": "string",
                    "description": "The first argument."
                },
                "str2": {
                    "type": "string",
                    "description": "The second argument."
                }
            },
            "required": ["str1", "str2"]
        }
    }
]


if __name__ == '__main__':


    client = OpenAI(api_key='sk-proj-48dZgyan5kkmwdDKXcoOT3BlbkFJ3j6Bnja0fl3wjaSuwjty',
                    organization='org-xI0JVzG8fKE2KZeoDnBisphz')

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are part of an Agentic workflow."},
            {"role": "user", "content": "Can you call user defined function 'concat' with parameters 'Hello' and 'Akhilesh'"},

        ],functions=functions
    )
    if response.choices[0].finish_reason == "function_call":
        function_call = response.choices[0].message.function_call
        function_name = function_call.name
        arguments = function_call.arguments


        args = json.loads(arguments)

        # Call the appropriate function
        if function_name == "concat":
            print("Open AI responded with function call")
            result = concat(args["str1"], args["str2"])

            # Send the result back to the user
            print(f"Function Result: {result}")


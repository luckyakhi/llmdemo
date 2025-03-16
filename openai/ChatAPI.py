import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()  # Looks for .env file by default

client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    organization=os.environ["OPENAI_ORG_ID"]
)

def generate_response(prompt, temperature=0.7):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=150
    )
    return response.choices[0].message.content

def print_response_with_factual_prompt(prompt="Explain quantum superposition in simple terms"):
    print(generate_response(prompt,temperature=0.0))

def print_response_with_creative_prompt(prompt="Write a short poem about AI ethics"):
    print(generate_response(prompt,temperature=0.7))

def print_response_with_tools(prompt="How old was Gandhi when India got independent"):
    prompt_with_tools = f""" 
    Tools:
    - calculator: This tool accepts math expressions and return results
    - search: This tool performs web search for a given query
    If you want tools to arrive at the answer list the tools to be used.
    Here is the input prompt : {prompt}
    """
    print(generate_response(prompt_with_tools))


if __name__ == '__main__':
    print_response_with_tools()





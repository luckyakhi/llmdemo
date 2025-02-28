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
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=150
    )
    return response.choices[0].message.content

# Example prompts
factual_prompt = "Explain quantum superposition in simple terms"
creative_prompt = "Write a short poem about AI ethics"
print(generate_response(factual_prompt, 0.2))
# Output: Clear, textbook-style explanation with consistent technical terms [2][5]

print(generate_response(creative_prompt, 0.7))
# Output: Structured poem with coherent theme and some creative metaphors [7][8]


import os
from typing import Literal
from IPython.display import Image, display
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain.model_utils import init_and_get_model
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
model = init_and_get_model()
@tool
def get_temperature(city: Literal["nyc", "sf"]):
    """Use this to get weather information."""
    if city == "nyc":
        return "13 degrees celsius"
    elif city == "blr":
        return "27 degrees celsius"
    else:
        raise AssertionError("Unknown city")
##Implement convert celsius to fahrenheit that can be used for tool calling
@tool
def convert_celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

tools = [get_temperature, convert_celsius_to_fahrenheit]
from langgraph.prebuilt import create_react_agent
graph = create_react_agent(model, tools=tools)
display(Image(graph.get_graph().draw_mermaid_png()))
#print(graph.get_graph().draw_mermaid())

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

inputs = {"messages": [("user", "what is the temperature in fahrenheit in nyc")]}
print_stream(graph.stream(inputs, stream_mode="values"))
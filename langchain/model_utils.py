import os

from dotenv import load_dotenv
from langchain_core.globals import set_debug, set_llm_cache
from langchain_openai import ChatOpenAI


def init_and_get_model(model_key = "OPENAI_MODEL"):
    set_debug(False)
    set_llm_cache(None)
    load_dotenv()
    model = ChatOpenAI(model=os.getenv(model_key), verbose=False)
    return model

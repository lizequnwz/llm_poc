"""Configuration for the LLM Apps Course"""
from types import SimpleNamespace
from dotenv import load_dotenv
import os

load_dotenv()

default_config = SimpleNamespace(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    vector_store_dir="llm-apps-course/vector_store",
    chat_prompt_dir="llm-apps-course/src",
    chat_temperature=0.3,
    max_fallback_retries=1,
    model_name="gpt-3.5-turbo",
    eval_model="gpt-3.5-turbo",
    #eval_artifact="darek/llmapps/generated_examples:v0",
)

default_config.__dict__
if __name__ == "__main__":
    print(default_config)
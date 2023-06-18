"""Configuration for the LLM Apps Course"""
from types import SimpleNamespace

default_config = SimpleNamespace(
    vector_store_dir="vector_store",
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
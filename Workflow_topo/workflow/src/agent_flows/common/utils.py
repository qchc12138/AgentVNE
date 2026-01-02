from langchain_openai import ChatOpenAI
import os


llm = ChatOpenAI(
    model="qwen3-30b-a3b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="any",  # if you prefer to pass api key in directly instaed of using env vars
    base_url="http://192.168.1.44:8021/v1",
)




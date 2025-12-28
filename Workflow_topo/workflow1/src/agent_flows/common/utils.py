from langchain_openai import ChatOpenAI
import os


llm = ChatOpenAI(
    # model="Qwen/Qwen2.5-7B-Instruct",
    # model="qwen3",
    model="qwen3-30b-a3b",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key="any",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="http://192.168.3.15:9997/v1", #qwen3
    base_url="http://192.168.1.44:8021/v1",
    # organization="...",
    # other params...
)




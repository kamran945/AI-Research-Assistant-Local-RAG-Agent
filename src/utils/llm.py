import os

from langchain_groq.chat_models import ChatGroq
from langchain_ollama import ChatOllama


if os.getenv("USE_GROQ") == "no":
    llm = ChatOllama(model=os.getenv("OLLAMA_CHAT_MODEL"), temperature=0.0)

    llm_json = ChatOllama(
        model=os.getenv("OLLAMA_CHAT_MODEL"), temperature=0.0, format="json"
    )
else:
    llm = ChatGroq(
        model="llama-3.1-70b-versatile", stop_sequences="[end]", temperature=0.0
    )
    llm_json = ChatGroq(
        model="llama-3.1-70b-versatile", stop_sequences="[end]", temperature=0.0
    )

"""Shared model configuration for all chains in the graph."""

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_openrouter import ChatOpenRouter

load_dotenv()

# Switch providers here: keep exactly one llm block uncommented.
# Restart the program after changing this configuration. Each chain adds its
# own prompt, tools, and output schema to this shared base model.

## Ollama Cloud: GLM
llm = ChatOllama(
    temperature=0.5,
    model="glm-5.3",
    reasoning=True,
    base_url="https://ollama.com",
)

## Ollama Cloud: Kimi
# llm = ChatOllama(
#     model="kimi-k3",
#     base_url="https://ollama.com",
#     temperature=0.5,
#     reasoning=True,
# )

# # OpenRouter
# llm = ChatOpenRouter(
#     model="openai/gpt-5.6-sol",
#     reasoning={"effort": "medium"},
#     timeout=120_000,  # ChatOpenRouter measures timeouts in milliseconds.
#     max_retries=2,
#     # Use OpenAI's endpoint through OpenRouter for strict tool schemas.
#     openrouter_provider={"only": ["openai"], "require_parameters": True},
# )

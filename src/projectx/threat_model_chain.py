import os
import datetime
from dotenv import load_dotenv
from projectx.schemas import ThreatModel

load_dotenv()

from langchain_ollama import ChatOllama
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.openai_tools import JsonOutputToolsParser, PydanticToolsParser

llm = ChatOllama(
    temperature=0.5,
    model="glm-5.3",
    num_ctx=8096,
    reasoning= True,
    base_url="https://ollama.com",
)

# llm = ChatOllama(
#     model="kimi-k3",
#     base_url="https://ollama.com",
#     temperature=0.5,
#     reasoning=True
# )
# OpenRouter alternative: swap the imports and llm blocks to use this provider.
# Uncomment the strict=True lines in the tool bindings only for OpenRouter.
# llm = ChatOpenRouter(
#     model="openai/gpt-5.6-sol",
#     reasoning={"effort": "medium"},
#     timeout=120_000,  # ChatOpenRouter measures timeouts in milliseconds.
#     max_retries=2,
#     # Use OpenAI's endpoint through OpenRouter for strict tool schemas.
#     openrouter_provider={"only": ["openai"], "require_parameters": True},
# )


threat_model_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are now a security researcher specialising in the domain of {discipline} that follows {programming_language} programming language.
            Current date: {time}
            1. Receive a codebase and take your time to understand it from a security standpoint. From there, fill up the threat model template.
                1.1. YOU MUST FILL UP AND FOLLOW THE TEMPLATE TO A T. THIS IS VERY IMPORTANT. FOLLOW IT AT ALL COSTS.
            2. Each field should be no more than 250 words. Do not feel obligated to hit 249 or 250 words flat.
                2.1. If you think a field needs 250 words then use 250 words. if you think a field only needs 10 words, then use 10 words.
            3. There is no rush to fill this up fast. It is more important to fill up the threat model with correct values than to be fast.
            """
        ),
        MessagesPlaceholder(variable_name="messages")
    ]
).partial(
    time=lambda:datetime.datetime.now().isoformat(),
)

structured_llm = llm.with_structured_output(
    ThreatModel,
    method="function_calling",
    include_raw=True,
)

threat_model_chain = threat_model_prompt_template | structured_llm


def generate_threat_model(context: str, language: str) -> ThreatModel:
    result = threat_model_chain.invoke({
        "discipline": "web",
        "programming_language": language,
        "messages": [
            HumanMessage(content=(
                "Return the report using the ThreatModel tool. "
                "Use the supplied evidence and explicitly mark unknowns. "
                "Do not claim that unperformed checks were completed.\n\n"
                + context
            ))
        ],
    })

    if result["parsing_error"] is not None:
        raise result["parsing_error"]

    report = result["parsed"]
    if not isinstance(report, ThreatModel):
        raise RuntimeError("The model did not return a ThreatModel.")

    return report

improve_instructions = """Revise the threat model that was made.
                        - You should use the previous critique to add important information to your answer.
                        - You MUST also use the previous critique to remove superfluous information and make SURE it is not more than 250 words for each field
                    """

if __name__ == "__main__":
    print(f"[INFO] NOT RUNNING Main.py.\n[Info] Now running {os.path.basename(__file__)}")
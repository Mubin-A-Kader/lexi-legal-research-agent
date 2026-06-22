"""Tool-calling agent. Claude decides which tools to call and how many times — the loop is
model-driven, not hard-coded. Built on LangChain so we get streaming callbacks for the UI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import config
from agent.llm import get_chat_model
from agent.tools import ALL_TOOLS
from agent.prompts import SYSTEM_PROMPT


def build_agent(callbacks=None, max_iterations: int = 12, model: str = None,
                temperature: float = 0) -> AgentExecutor:
    llm = get_chat_model(model or config.AGENT_MODEL, max_tokens=4096, temperature=temperature)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history", optional=True),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, ALL_TOOLS, prompt)
    return AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        callbacks=callbacks,
        max_iterations=max_iterations,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
        verbose=True,
    )


def run(query: str, callbacks=None) -> dict:
    """Convenience for CLI/eval. Returns {output, intermediate_steps}."""
    executor = build_agent(callbacks=callbacks)
    return executor.invoke({"input": query})


if __name__ == "__main__":
    q = " ".join(sys.argv[1:]) or "Which judgments involve commercial vehicles?"
    result = run(q)
    print("\n=== FINAL ===\n", result["output"])

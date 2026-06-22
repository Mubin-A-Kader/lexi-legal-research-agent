"""Provider-agnostic LLM access. One switch (config.LLM_PROVIDER) flips OpenAI <-> Anthropic.

- get_chat_model(): LangChain chat model for the tool-calling agent.
- complete(): single completion for analyze_precedent / LLM-judge evals.

Default provider is OpenAI (gpt-4o-mini) — cheapest with reliable tool-calling. Set
OPENAI_BASE_URL to reuse this code against any OpenAI-compatible API (e.g. DeepSeek).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def get_chat_model(model: str = None, max_tokens: int = 4096, temperature: float = 0):
    model = model or config.AGENT_MODEL
    if config.LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic

        use_thinking = any(m in model for m in
                           ("opus-4-6", "opus-4-7", "opus-4-8", "sonnet-4-6", "fable-5"))
        kwargs = dict(model=model, max_tokens=max_tokens)
        if use_thinking:
            kwargs["thinking"] = {"type": "adaptive"}  # temperature unsupported with thinking
        else:
            kwargs["temperature"] = temperature
        return ChatAnthropic(**kwargs)

    from langchain_openai import ChatOpenAI

    kwargs = dict(model=model, temperature=temperature, max_tokens=max_tokens)
    if config.OPENAI_BASE_URL:
        kwargs["base_url"] = config.OPENAI_BASE_URL
    return ChatOpenAI(**kwargs)


def complete(prompt: str, model: str = None, max_tokens: int = 1024) -> str:
    """Single-shot text completion. Returns the model's text."""
    model = model or config.EVAL_MODEL
    if config.LLM_PROVIDER == "anthropic":
        from anthropic import Anthropic

        resp = Anthropic().messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()

    from openai import OpenAI

    client = OpenAI(base_url=config.OPENAI_BASE_URL) if config.OPENAI_BASE_URL else OpenAI()
    resp = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip()

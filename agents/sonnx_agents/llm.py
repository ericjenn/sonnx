"""The chat model, and the one place a response is turned into text.

The model is any endpoint that speaks the Anthropic Messages API — the default is the
one this repository's `ANTHROPIC_BASE_URL` names, which may be a local gateway to a
non-Anthropic model.  Such models answer with *thinking* blocks before the text, so the
only safe reading of a response is "join the text blocks, drop the rest", which is what
``text_of`` does.  Nothing else in the system touches ``message.content``.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .config import Settings


def build_llm(settings: Settings, max_tokens: int | None = None,
              thinking: str = "auto") -> ChatAnthropic:
    kwargs = {
        "model": settings.model,
        "max_tokens": max_tokens or settings.max_tokens,
        "temperature": settings.temperature,
        "timeout": settings.timeout,
        "max_retries": 2,
    }
    if settings.base_url:
        kwargs["base_url"] = settings.base_url
    if settings.api_key:
        kwargs["api_key"] = settings.api_key
    if thinking == "off":
        # a model that deliberates before it answers can spend its whole output budget on
        # the deliberation and return no answer at all — measured here at 61 000 characters
        # of thinking and an empty reply for a whole-document compliance review.  For a
        # bounded verdict the deliberation is turned off rather than raced.
        kwargs["thinking"] = {"type": "disabled"}
    return ChatAnthropic(**kwargs)


def text_of(message: AIMessage) -> str:
    """The text of an assistant message, thinking blocks and all else dropped."""
    content = message.content
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content or []:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") in (None, "text"):
            parts.append(str(block.get("text", "")))
    return "\n".join(parts).strip()


def ask(settings: Settings, system: str, user: str, *, max_tokens: int | None = None,
        thinking: str = "auto") -> str:
    """One round trip: a system message and a user message, the text back.

    A model that thinks before it answers can spend its whole budget on the thinking and
    return no text at all; that is worth one retry with twice the budget rather than an
    empty artifact and a wasted round.  That retry is not enough on its own: measured on
    2026-10-04, a blind implementer answered with thinking and no text twice, and the empty
    answer was written to disk as the implementation of **Asin** — which is why the node
    validates what it gets (see `graph.implement`).  When the empty answer is the *habit* of
    the task rather than an accident, the caller turns the deliberation off instead
    (``thinking="off"``).
    """
    budget = max_tokens or settings.max_tokens
    llm = build_llm(settings, budget, thinking)
    text = text_of(llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]))
    if not text:
        llm = build_llm(settings, budget * 2, thinking)
        text = text_of(llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user + "\n\nAnswer with the artifact itself, now."),
        ]))
    return text

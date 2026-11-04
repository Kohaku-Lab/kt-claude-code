"""Fit the inline images of a transcript to the API's per-request dimension limits."""

from typing import Any

from kohakuterrarium.llm.anthropic.images import prepare_anthropic_images

_COUNTED = {"type": "image", "source": {"type": "url", "url": ""}}


def count_images(content: Any) -> int:
    """Images in message ``content``, including those nested in tool results."""
    if not isinstance(content, list):
        return 0
    total = 0
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "image":
            total += 1
        elif block.get("type") == "tool_result":
            total += count_images(block.get("content"))
    return total


def fit_images(
    messages: list[dict[str, Any]], *, extra: int = 0
) -> list[dict[str, Any]]:
    """``messages`` with oversized images resized.

    ``extra`` counts images sent outside ``messages`` (the step's prompt): the
    request-wide limit depends on every image of the step.
    """
    padding = [{"role": "user", "content": [dict(_COUNTED) for _ in range(extra)]}]
    fitted = prepare_anthropic_images([*messages, *(padding if extra else [])])
    return fitted[: len(messages)]

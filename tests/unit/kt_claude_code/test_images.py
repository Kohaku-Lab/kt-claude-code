"""Unit tests for ``kt_claude_code.images`` and its use in the transcript."""

import base64
import io

from PIL import Image

from kt_claude_code.images import count_images, fit_images
from kt_claude_code.transcript import plan_transcript, transcript_entries


def _png(width: int, height: int) -> dict:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (10, 20, 30)).save(buffer, format="PNG")
    data = base64.b64encode(buffer.getvalue()).decode("ascii")
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": data},
    }


def _size(block: dict) -> tuple[int, int]:
    raw = base64.b64decode(block["source"]["data"])
    return Image.open(io.BytesIO(raw)).size


def _sizes(messages: list[dict]) -> set:
    return {
        _size(b) for m in messages for b in m["content"] if b.get("type") == "image"
    }


def _user(*blocks: dict) -> dict:
    return {"role": "user", "content": list(blocks)}


class TestFitImages:
    def test_a_request_of_21_images_caps_each_at_2000px(self):
        messages = [_user(*[_png(2100, 1350) for _ in range(21)])]
        fitted = fit_images(messages)
        assert {_size(b) for b in fitted[0]["content"]} == {(2000, 1286)}

    def test_20_images_keep_their_size(self):
        messages = [_user(*[_png(2100, 1350) for _ in range(20)])]
        assert fit_images(messages) is not None
        assert _size(fit_images(messages)[0]["content"][0]) == (2100, 1350)

    def test_images_sent_outside_the_messages_count_toward_the_limit(self):
        messages = [_user(*[_png(2100, 1350) for _ in range(20)])]
        fitted = fit_images(messages, extra=1)
        assert len(fitted) == 1
        assert _size(fitted[0]["content"][0]) == (2000, 1286)

    def test_images_inside_tool_results_are_counted_and_resized(self):
        result = {
            "type": "tool_result",
            "tool_use_id": "t",
            "content": [_png(2100, 1350)],
        }
        messages = [_user(*[_png(100, 100) for _ in range(20)]), _user(result)]
        assert count_images(messages[1]["content"]) == 1
        fitted = fit_images(messages)
        assert _size(fitted[1]["content"][0]["content"][0]) == (2000, 1286)


class TestTranscriptImages:
    def test_history_and_replayed_entries_are_resized(self):
        big = [
            {
                "type": "image_url",
                "image_url": {
                    "url": "data:image/png;base64," + _png(2100, 1350)["source"]["data"]
                },
            }
            for _ in range(21)
        ]
        messages = [
            {"role": "user", "content": big},
            {"role": "assistant", "content": "ok"},
        ]
        plan = plan_transcript(messages)
        assert _sizes(plan.history) == {(2000, 1286)}
        entries = transcript_entries(
            plan.history, session_id="s", cwd=".", version="1", model="m"
        )
        sizes = _sizes([e["message"] for e in entries])
        assert sizes == {(2000, 1286)}

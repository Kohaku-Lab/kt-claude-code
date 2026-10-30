"""Unit tests for ``kt_claude_code.errors``."""

import pytest

from kohakuterrarium.errors import LLMNotConfiguredError
from kohakuterrarium.llm.recovery.errors import ErrorClass

from kt_claude_code.errors import ClaudeCodeError, classify, error_from_result


@pytest.mark.parametrize(
    "kind, text, expected",
    [
        ("rate_limit", "limit reached", ErrorClass.RATE_LIMIT),
        ("server_error", "overloaded", ErrorClass.SERVER),
        ("invalid_request", "Prompt is too long", ErrorClass.OVERFLOW),
        ("unknown", "connection reset by peer", ErrorClass.TRANSIENT),
        (None, "something odd", ErrorClass.UNKNOWN),
    ],
)
def test_recoverable_kinds(kind, text, expected):
    err = error_from_result(kind, text)
    assert isinstance(err, ClaudeCodeError)
    assert err.error_class is expected
    assert str(err) == text


def test_blocking_limit_is_overflow_whatever_the_text():
    err = error_from_result("invalid_request", "", terminal_reason="blocking_limit")
    assert err.error_class is ErrorClass.OVERFLOW


@pytest.mark.parametrize(
    "kind", ["authentication_failed", "billing_error", "model_not_found"]
)
def test_setup_problems_are_configuration_errors(kind):
    err = error_from_result(kind, "details here")
    assert isinstance(err, LLMNotConfiguredError)
    assert classify(err) is ErrorClass.USER_ERROR


def test_sign_in_message_tells_the_user_to_log_in_themselves():
    err = error_from_result("authentication_failed", "Not logged in")
    assert "/login" in str(err) and "never handles credentials" in str(err)


def test_classify_other_exceptions_with_core_rules():
    assert classify(TimeoutError()) is ErrorClass.TRANSIENT
    assert classify(ClaudeCodeError("x", ErrorClass.SERVER)) is ErrorClass.SERVER

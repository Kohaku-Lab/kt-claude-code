"""Map Claude Code failures to KohakuTerrarium's typed errors and recovery classes."""

from kohakuterrarium.errors import LLMNotConfiguredError
from kohakuterrarium.llm.recovery.errors import ErrorClass, classify_openai_error

LOGIN_HINT = (
    "Sign in to Claude Code yourself: run `claude`, then `/login`. "
    "This backend never handles credentials."
)


class ClaudeCodeError(RuntimeError):
    """A failed model step, carrying the recovery class KT should apply."""

    def __init__(
        self, message: str, error_class: ErrorClass, status_code: int | None = None
    ):
        super().__init__(message)
        self.error_class = error_class
        self.status_code = status_code


def error_from_result(
    kind: str | None,
    text: str,
    *,
    status_code: int | None = None,
    terminal_reason: str | None = None,
) -> Exception:
    """The exception for an error result reported by the CLI.

    ``kind`` is the SDK's ``AssistantMessage.error`` value, when present.
    """
    message = text.strip() or kind or "Claude Code returned an error"
    match kind:
        case "authentication_failed":
            return LLMNotConfiguredError(f"Claude Code is not signed in. {LOGIN_HINT}")
        case "billing_error" | "model_not_found":
            return LLMNotConfiguredError(f"Claude Code: {message}")
        case "rate_limit":
            return ClaudeCodeError(message, ErrorClass.RATE_LIMIT, status_code)
        case "server_error":
            return ClaudeCodeError(message, ErrorClass.SERVER, status_code)
    if terminal_reason == "blocking_limit":
        return ClaudeCodeError(message, ErrorClass.OVERFLOW, status_code)
    # An error result is a response, never a transport failure: status 0 makes
    # core classify unmatched text as UNKNOWN instead of a retryable TRANSIENT.
    probe = RuntimeError(message)
    probe.status_code = status_code or 0  # type: ignore[attr-defined]
    return ClaudeCodeError(message, classify_openai_error(probe), status_code)


def classify(exc: BaseException) -> ErrorClass:
    """Recovery class for any exception raised during a step."""
    if isinstance(exc, ClaudeCodeError):
        return exc.error_class
    if isinstance(exc, LLMNotConfiguredError):
        return ErrorClass.USER_ERROR
    return classify_openai_error(exc)

import json
import logging
import re
import threading
import time

import openai

log = logging.getLogger(__name__)

# Transient failures worth retrying (the OpenAI SDK also retries these twice itself).
RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

# Optional request parameters. Some models reject them (e.g. reasoning models
# reject temperature; some OpenRouter models lack JSON mode); they are dropped
# per model on the first 400 error and the call is retried without them.
OPTIONAL_PARAMS = {
    "temperature": 0,
    "response_format": {"type": "json_object"},
}


def parse_json_object(raw):
    """Parse one JSON object, tolerating Markdown fences or text around it."""
    text = (raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found")
    value = json.loads(text[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


class BaseAgent:
    """Common OpenAI-compatible client wrapper for all agents."""

    max_attempts = 4
    _dropped_params = {}  # model -> optional params that model rejected
    calls_made = 0  # LLM requests sent by all agents (shown in the run summary)
    _lock = threading.Lock()

    def __init__(self, client, model):
        self.client = client
        self.model = model

    def ask_json(self, system_prompt, user_prompt):
        """Call the model with explicit system/user prompts and parse JSON.

        Retries transient API errors with backoff, and asks the model to repair
        its reply when it is not a valid JSON object.
        """
        messages = [
            {
                "role": "system",
                "content": (
                    system_prompt
                    + "\nReturn exactly one valid JSON object. "
                    "Do not include Markdown fences or commentary."
                ),
            },
            {"role": "user", "content": user_prompt},
        ]
        dropped = self._dropped_params.setdefault(self.model, set())
        last_error = None

        for attempt in range(self.max_attempts):
            params = {
                key: value
                for key, value in OPTIONAL_PARAMS.items()
                if key not in dropped
            }
            try:
                with BaseAgent._lock:
                    BaseAgent.calls_made += 1
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **params,
                )
            except openai.BadRequestError as exc:
                if not params:
                    raise
                message = str(exc).lower()
                rejected = [k for k in params if k in message] or list(params)
                log.warning(
                    "Model %s doesn't accept %s; retrying without it.",
                    self.model, " / ".join(rejected),
                )
                dropped.update(rejected)
                continue
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                delay = min(30, 2 ** attempt)
                log.warning("%s: temporary API problem (%s); retrying in %ss.",
                            type(self).__name__, type(exc).__name__, delay)
                time.sleep(delay)
                continue

            choice = response.choices[0] if response.choices else None
            raw = (choice.message.content if choice else "") or ""
            try:
                return parse_json_object(raw)
            except ValueError as exc:  # includes json.JSONDecodeError
                last_error = exc
                truncated = getattr(choice, "finish_reason", None) == "length"
                feedback = (
                    "Your reply was cut off before the JSON ended. Return a "
                    "shorter but complete JSON object."
                    if truncated
                    else f"That was not one valid JSON object ({exc}). "
                    "Return only the corrected JSON object."
                )
                messages = messages[:2] + [
                    {"role": "assistant", "content": raw[:4000]},
                    {"role": "user", "content": feedback},
                ]
                log.warning("%s: reply was not valid JSON (attempt %d of %d); "
                            "asking the model to fix it.",
                            type(self).__name__, attempt + 1, self.max_attempts)

        raise ValueError(
            f"{type(self).__name__} failed after {self.max_attempts} "
            f"attempts: {last_error}"
        )

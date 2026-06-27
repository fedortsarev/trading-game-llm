"""Provider abstraction: one interface, many model backends.

Every client forces a single tool call named `submit_action` whose input schema is
the (inlined) Action JSON Schema, and returns the parsed tool input as `tool_input`.
This gives structured output + provider-side validation for free; the worker still
hard-validates with Pydantic. Provider SDKs are imported lazily so the engine and
offline tests never need them or an API key.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

TOOL_NAME = "submit_action"


@dataclass
class LLMResult:
    tool_input: dict | None  # parsed arguments of the forced tool call, or None
    raw_text: str = ""  # any free text the model emitted alongside the tool call
    model: str = ""
    usage: dict = field(default_factory=dict)
    error: str | None = None  # set when the call failed or returned no tool call
    latency_s: float = 0.0


class LLMClient(Protocol):
    def submit(self, system: str, user: str, tool_schema: dict, *,
               timeout: float) -> LLMResult:
        ...


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------
class AnthropicClient:
    def __init__(self, model: str, *, api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024):
        import anthropic  # lazy

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    def submit(self, system: str, user: str, tool_schema: dict, *,
               timeout: float) -> LLMResult:
        t0 = time.monotonic()
        try:
            resp = self._client.messages.create(
                model=self.model,
                system=system,
                messages=[{"role": "user", "content": user}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tools=[{
                    "name": TOOL_NAME,
                    "description": "Submit your action for this round.",
                    "input_schema": tool_schema,
                }],
                tool_choice={"type": "tool", "name": TOOL_NAME},
                timeout=timeout,
            )
        except Exception as e:  # network/timeout/api error -> caller degrades to pass
            return LLMResult(None, error=f"{type(e).__name__}: {e}", model=self.model,
                             latency_s=time.monotonic() - t0)

        tool_input, text = None, []
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == TOOL_NAME:
                tool_input = dict(block.input)
            elif getattr(block, "type", None) == "text":
                text.append(block.text)
        usage = {"input_tokens": getattr(resp.usage, "input_tokens", None),
                 "output_tokens": getattr(resp.usage, "output_tokens", None)}
        return LLMResult(
            tool_input=tool_input,
            raw_text="".join(text),
            model=self.model,
            usage=usage,
            error=None if tool_input is not None else "no tool_use block returned",
            latency_s=time.monotonic() - t0,
        )


# --------------------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------------------
class OpenAIClient:
    def __init__(self, model: str, *, api_key: str | None = None,
                 temperature: float = 0.0, max_tokens: int = 1024):
        import openai  # lazy

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    def submit(self, system: str, user: str, tool_schema: dict, *,
               timeout: float) -> LLMResult:
        t0 = time.monotonic()
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tools=[{
                    "type": "function",
                    "function": {
                        "name": TOOL_NAME,
                        "description": "Submit your action for this round.",
                        "parameters": tool_schema,
                    },
                }],
                tool_choice={"type": "function", "function": {"name": TOOL_NAME}},
                timeout=timeout,
            )
        except Exception as e:
            return LLMResult(None, error=f"{type(e).__name__}: {e}", model=self.model,
                             latency_s=time.monotonic() - t0)

        msg = resp.choices[0].message
        tool_input, error = None, None
        calls = msg.tool_calls or []
        if calls:
            try:
                tool_input = json.loads(calls[0].function.arguments)
            except json.JSONDecodeError as e:
                error = f"tool arguments not valid JSON: {e}"
        else:
            error = "no tool call returned"
        usage = {"input_tokens": getattr(resp.usage, "prompt_tokens", None),
                 "output_tokens": getattr(resp.usage, "completion_tokens", None)}
        return LLMResult(
            tool_input=tool_input,
            raw_text=msg.content or "",
            model=self.model,
            usage=usage,
            error=error,
            latency_s=time.monotonic() - t0,
        )


# --------------------------------------------------------------------------
# Mock — for offline tests and demos (no SDK, no network, no key)
# --------------------------------------------------------------------------
class MockLLMClient:
    """Returns scripted results. `script` is either a list (consumed in order) or a
    callable(system, user, tool_schema) -> LLMResult."""

    def __init__(self, script: list[LLMResult] | Callable[..., LLMResult],
                 model: str = "mock"):
        self.model = model
        self._script = script
        self._i = 0
        self.calls: list[dict] = []

    def submit(self, system: str, user: str, tool_schema: dict, *,
               timeout: float) -> LLMResult:
        self.calls.append({"system": system, "user": user, "tool_schema": tool_schema})
        if callable(self._script):
            res = self._script(system, user, tool_schema)
        else:
            res = self._script[min(self._i, len(self._script) - 1)]
            self._i += 1
        if not res.model:
            res.model = self.model
        return res

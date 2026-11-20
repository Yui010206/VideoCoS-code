"""Model API client (ported unchanged from VideoCoS infer/provider.py) and a per-run cost meter.

Chat Completions / Responses HTTP transport, typed provider failures and retry timing.

One ``complete`` call is one HTTP attempt. ``ProviderError.unbilled`` marks a
failure that certainly produced no billable work (request refused before it
was processed); timeouts, broken reads and 5xx stay uncertain.

With ``api_style: "responses"`` the client speaks the Responses API but keeps
the Chat Completions shape at its boundary, so the agent and ledger are shared.
Requests are stateless (``store: false``); reasoning is replayed encrypted.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import http.client
import json
import math
import socket
import urllib.error
import urllib.parse
import urllib.request



def _tool_choice(tools, force_final=False):
    """Require the sole submission function; keep scene tools automatic."""
    if force_final:
        return "none"
    if len(tools) == 1:
        name = tools[0].get("function", {}).get("name")
        if name in ("submit_answer", "submit_update"):
            return {"type": "function", "function": {"name": name}}
    return "auto"


class ProviderError(RuntimeError):
    def __init__(self, message, *, retryable=False, status_code=None, retry_after=None, unbilled=False):
        super().__init__(message)
        self.retryable = bool(retryable)
        self.status_code = status_code
        self.retry_after = retry_after
        self.unbilled = bool(unbilled)


def retry_after_seconds(value):
    """Parse Retry-After without exposing response headers or bodies in logs."""
    if not value:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)
            seconds = (date - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0., seconds) if math.isfinite(seconds) else None


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 2
    base_seconds: float = 1.
    maximum_seconds: float = 30.

    def __post_init__(self):
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= 2:
            raise ValueError('max_retries must be an integer in 0..2')
        if (not math.isfinite(self.base_seconds) or not math.isfinite(self.maximum_seconds)
                or not 0 <= self.base_seconds <= self.maximum_seconds <= 30):
            raise ValueError('Retry timing must satisfy 0 <= base <= maximum <= 30 seconds')

    def delay(self, retry_index, retry_after=None):
        """Exponential backoff, at least Retry-After, never above the local cap."""
        delay = self.base_seconds * 2 ** retry_index
        if retry_after is not None and math.isfinite(retry_after):
            delay = max(delay, retry_after)
        return min(delay, self.maximum_seconds)


def _responses_body(model, messages, tools, tool_choice, max_tokens, reasoning_effort):
    """Translate Chat Completions messages and tools into one Responses request."""
    instructions, items = [], []
    for m in messages:
        if m["role"] == "system":
            instructions.append(m["content"])
        elif m["role"] == "tool":
            items.append({"type": "function_call_output", "call_id": m["tool_call_id"], "output": m["content"]})
        elif m["role"] == "assistant":
            items.extend(m.get("reasoning_items") or [])
            if m.get("content"):
                items.append({"role": "assistant", "content": m["content"]})
            items.extend({"type": "function_call", "call_id": c["id"], "name": c["function"]["name"],
                          "arguments": c["function"]["arguments"]} for c in m.get("tool_calls") or [])
        else:
            items.append({"role": m["role"], "content": m["content"]})
    body = {"model": model, "input": items, "max_output_tokens": max_tokens,
            "store": False, "include": ["reasoning.encrypted_content"]}
    if instructions:
        body["instructions"] = "\n\n".join(instructions)
    if tools:
        # Chat Completions schemas are not strict-mode schemas; keep their semantics.
        body["tools"] = [{"type": "function", **t["function"], "strict": False} for t in tools]
        body["tool_choice"] = ({"type": "function", "name": tool_choice["function"]["name"]}
                               if isinstance(tool_choice, dict) else tool_choice)
    if reasoning_effort is not None:
        body["reasoning"] = {"effort": reasoning_effort}
    return body


def _chat_payload(response):
    """Map a Responses result onto the Chat Completions fields the agent and ledger read."""
    if not isinstance(response, dict) or not isinstance(response.get("output"), list):
        raise ProviderError("Model API returned no Responses output.")
    if response.get("error"):
        raise ProviderError("Model API returned a failed response.")
    text, refusal, calls, reasoning = [], [], [], []
    for item in response["output"]:
        if item.get("type") == "reasoning":
            reasoning.append(item)
        elif item.get("type") == "function_call":
            calls.append({"id": item["call_id"], "type": "function",
                          "function": {"name": item["name"], "arguments": item["arguments"]}})
        elif item.get("type") == "message":
            for part in item.get("content") or []:
                if part.get("type") == "output_text":
                    text.append(part.get("text", ""))
                elif part.get("type") == "refusal":
                    refusal.append(part.get("refusal", ""))
    if response.get("status") == "incomplete":
        reason = (response.get("incomplete_details") or {}).get("reason")
        finish = "length" if reason == "max_output_tokens" else "content_filter"
    else:
        finish = "tool_calls" if calls else "stop"
    message = {"role": "assistant", "content": "".join(text) or None}
    if refusal:
        message["refusal"] = "".join(refusal)
    if calls:
        message["tool_calls"] = calls
    if reasoning:
        message["reasoning_items"] = reasoning
    raw = response.get("usage") or {}
    usage = {"prompt_tokens": raw.get("input_tokens"), "completion_tokens": raw.get("output_tokens"),
             "total_tokens": raw.get("total_tokens")}
    if "reasoning_tokens" in (raw.get("output_tokens_details") or {}):
        usage["completion_tokens_details"] = {"reasoning_tokens": raw["output_tokens_details"]["reasoning_tokens"]}
    return {"id": response.get("id"), "model": response.get("model"), "usage": usage,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}]}


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class ChatClient:
    """Small injectable HTTP client; credentials never enter saved payloads."""

    def __init__(self, api_config):
        self.api_key = api_config.get("api_key", "")
        self.model = api_config.get("model", "")
        if not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValueError("API key is blank. Fill api.api_key in config.json or set the configured API-key environment variable before API mode.")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("API model is blank. Set api.model to a model that supports tool calling.")
        base_url = api_config.get("base_url", "https://api.openai.com/v1")
        parsed = urllib.parse.urlsplit(base_url)
        if (parsed.scheme not in ("http", "https") or not parsed.netloc
                or parsed.query or parsed.fragment or parsed.username or parsed.password):
            raise ValueError("API base_url must be an HTTP(S) API root without credentials, query or fragment.")
        if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
            raise ValueError("A remote API endpoint must use HTTPS.")
        self.api_style = api_config.get("api_style", "chat")
        if self.api_style not in ("chat", "responses"):
            raise ValueError('api_style must be "chat" or "responses".')
        self.endpoint = base_url.rstrip("/") + ("/responses" if self.api_style == "responses" else "/chat/completions")
        self.timeout = float(api_config.get("timeout", 120))
        self.max_output_tokens = int(api_config.get("max_output_tokens", 4096))
        self.max_final_output_tokens = int(api_config.get("max_final_output_tokens", self.max_output_tokens))
        # Optional; e.g. "none" for models that reject function tools with reasoning on Chat Completions.
        self.reasoning_effort = api_config.get("reasoning_effort")
        # Optional sampling temperature (e.g. 0.6 for Qwen thinking); omitted means the server default.
        self.temperature = api_config.get("temperature")
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("API timeout must be positive and finite.")
        if not 1 <= self.max_output_tokens <= 131072:
            raise ValueError("max_output_tokens must be between 1 and 131072.")
        if not 1 <= self.max_final_output_tokens <= 131072:
            raise ValueError("max_final_output_tokens must be between 1 and 131072.")

    def request_output_token_limit(self, tools, force_final=False):
        """Optional extra capacity applies only to the sole forced submission."""
        return (self.max_final_output_tokens if isinstance(_tool_choice(tools, force_final), dict)
                else self.max_output_tokens)

    def complete(self, messages, tools, force_final=False, max_tokens=None):
        """``max_tokens`` caps this one request below the configured limit (length-recovery turns)."""
        limit = self.request_output_token_limit(tools, force_final)
        if max_tokens is not None:
            limit = min(limit, int(max_tokens))
        if self.api_style == "responses":
            body = _responses_body(self.model, messages, tools, _tool_choice(tools, force_final) if tools else None,
                                   limit, self.reasoning_effort)
        else:
            body = {"model": self.model, "messages": messages, "max_completion_tokens": limit}
            if tools:
                body.update(tools=tools, tool_choice=_tool_choice(tools, force_final))
            if self.reasoning_effort is not None:
                body["reasoning_effort"] = self.reasoning_effort
            if self.temperature is not None:
                body["temperature"] = self.temperature
        request = urllib.request.Request(
            self.endpoint, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"})
        try:
            with urllib.request.build_opener(_NoRedirect).open(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except urllib.error.HTTPError as exc:
            # Provider error bodies can echo request credentials or image data.
            error = ProviderError(f"Model API returned HTTP {exc.code}. Check endpoint, model access and API-style compatibility.",
                                  retryable=exc.code == 429 or 500 <= exc.code <= 599,
                                  status_code=exc.code, unbilled=400 <= exc.code <= 499,
                                  retry_after=retry_after_seconds(exc.headers.get('Retry-After') if exc.headers else None))
            exc.close()
            raise error from None
        except urllib.error.URLError as exc:
            # Refused connections and DNS failures never reached the provider.
            refused = isinstance(exc.reason, (ConnectionRefusedError, socket.gaierror))
            raise ProviderError("Could not connect to the model API.", retryable=True, unbilled=refused) from None
        except (OSError, http.client.HTTPException):
            raise ProviderError("Model API connection failed or timed out after the request was sent.", retryable=True) from None
        except (ValueError, UnicodeError):
            raise ProviderError("Model API returned invalid JSON.") from None
        if self.api_style == "responses":
            payload = _chat_payload(payload)
        if not isinstance(payload, dict) or not isinstance(payload.get("choices"), list) or not payload["choices"]:
            raise ProviderError("Model API returned no Chat Completions choices.")
        return payload


# --------------------------------------------------------------------------- cost
PRICES = {  # USD per million tokens; output includes thinking. Local models cost nothing but are still counted.
    "gemini-3.8-flash": (0.75, 3.75),
    "gpt-5.6-luna": (0.20, 1.20),
}


class BudgetExceeded(RuntimeError):
    """The next request could cross the run's cap; it was not sent."""


class MeteredClient:
    """Wraps a ChatClient: every request is priced at its worst case BEFORE it is sent and refused if
    the run cap could be crossed, then charged at the usage the provider reports. Retries 429/5xx and
    transport errors twice with backoff. One meter is shared by every task in a run."""

    def __init__(self, clients, cap_usd, ledger_path, local=False):
        """``clients``: {"query": ChatClient, "update": ChatClient}, the same model with each kind's limits."""
        import threading
        self.clients, self.cap, self.ledger = dict(clients), float(cap_usd), ledger_path
        model = next(iter(self.clients.values())).model
        self.prices = (0.0, 0.0) if local else PRICES.get(model)
        if self.prices is None:
            raise ValueError(f"no price for {model!r}; add it to PRICES or pass local=True")
        self.spent = 0.0
        self.requests = 0
        self._lock = threading.Lock()
        self.retry = RetryPolicy()

    def _cost(self, prompt, completion):
        return (prompt * self.prices[0] + completion * self.prices[1]) / 1e6

    def for_task(self, task_id, kind):
        meter, client = self, self.clients[kind]

        class Bound:
            def request_output_token_limit(self, tools, force_final=False):
                return client.request_output_token_limit(tools, force_final)

            def complete(self, messages, tools, force_final=False, max_tokens=None):
                return meter.complete(client, messages, tools, force_final, max_tokens=max_tokens, task_id=task_id)
        return Bound()

    def complete(self, client, messages, tools, force_final=False, max_tokens=None, task_id=None):
        import time
        size = len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode())
        limit = client.request_output_token_limit(tools, force_final)
        if max_tokens is not None:
            limit = min(limit, int(max_tokens))
        worst = self._cost(size + 1024 + 64 * len(messages), limit)   # bytes bound tokens from above
        for attempt in range(self.retry.max_retries + 1):
            with self._lock:
                if self.spent + worst > self.cap:
                    raise BudgetExceeded(f"run cap ${self.cap:.2f}: spent ${self.spent:.4f}, this request could cost "
                                         f"${worst:.4f}; not sent")
                self.spent += worst                     # reserved until the usage is known
                self.requests += 1
            try:
                payload = client.complete(messages, tools, force_final, max_tokens=max_tokens)
            except ProviderError as exc:
                with self._lock:
                    if exc.unbilled:
                        self.spent -= worst             # certainly not billed: release the reservation
                self._log(task_id, None, worst if not exc.unbilled else 0.0, f"error {exc.status_code}")
                if not exc.retryable or attempt >= self.retry.max_retries:
                    raise
                time.sleep(self.retry.delay(attempt, exc.retry_after))
                continue
            usage = payload.get("usage") or {}
            prompt, completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
            total = usage.get("total_tokens")
            if isinstance(prompt, int) and isinstance(total, int):
                # Gemini bills thinking as output but leaves it out of completion_tokens; it is in total_tokens.
                completion = max(total - prompt, completion if isinstance(completion, int) else 0)
            if isinstance(prompt, int) and isinstance(completion, int):
                actual = self._cost(prompt, completion)
                with self._lock:
                    self.spent += actual - worst
            else:
                actual = worst                          # usage missing: keep the worst case
            self._log(task_id, usage, actual, "ok")
            return payload

    def _log(self, task_id, usage, cost, status):
        with self._lock, open(self.ledger, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"task": task_id, "status": status, "usage": usage,
                                     "cost_usd": round(cost, 6), "spent_usd": round(self.spent, 6)}) + "\n")

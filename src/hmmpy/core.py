"""
HmmPy
=====

A lightweight, provider-agnostic probabilistic decision session for any
OpenAI-compatible Chat Completions endpoint.

Developer: Pouriya Khalilian
Version: 0.3.3
License: MIT

HmmPy is intentionally small: put this file next to a notebook and import it.

Important note on probabilities
-------------------------------
HmmPy estimates uncertainty by repeated model sampling and aggregation. The
returned probabilities are model-derived estimates and are NOT guaranteed to be
statistically calibrated probabilities. Version 0.2 adds optional confidence
gating/abstention. For calibrated probabilities, evaluate the system on labeled
data and add a calibration layer.

Example
-------
>>> from hmmpy import HmmSession
>>> hmm = HmmSession(
...     base_url="http://localhost:1234/v1",
...     api_key="local",
...     model="my-model",
... )
>>> hmm.set_system_prompt("Be careful and conservative.")
>>> result = hmm.boolean("Is the supplied idea internally consistent?", samples=5)
>>> print(result.value, result.probability)
>>> print(hmm.usage())
"""

from __future__ import annotations

import copy
import json
import math
import re
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, Union

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "HmmPy requires the `openai` package. Install it with: pip install -U openai"
    ) from exc


__title__ = "HmmPy"
__version__ = "0.3.3"
__author__ = "Pouriya Khalilian"
__license__ = "MIT"


class HmmPyError(Exception):
    """Base exception for HmmPy."""


class HmmParseError(HmmPyError):
    """Raised when a provider response cannot be parsed into a decision."""


class HmmConfigurationError(HmmPyError):
    """Raised when the session configuration is incomplete or invalid."""


@dataclass
class Usage:
    """Token usage for one call or an accumulated session."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cached_tokens=self.cached_tokens + other.cached_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            requests=self.requests + other.requests,
        )

    def to_dict(self) -> Dict[str, int]:
        return asdict(self)


@dataclass
class Pricing:
    """
    Price per 1,000,000 tokens.

    `cached_input` and `reasoning` are optional because not every provider
    exposes or prices those token categories separately.
    """

    input: float = 0.0
    output: float = 0.0
    cached_input: Optional[float] = None
    reasoning: Optional[float] = None
    currency: str = "USD"

    def estimate(self, usage: Usage) -> float:
        uncached_prompt = max(0, usage.prompt_tokens - usage.cached_tokens)
        input_cost = uncached_prompt / 1_000_000 * self.input

        cached_rate = self.cached_input if self.cached_input is not None else self.input
        cached_cost = usage.cached_tokens / 1_000_000 * cached_rate

        # reasoning_tokens are usually included inside completion_tokens by providers,
        # so do not double-charge unless an explicit reasoning rate is supplied.
        if self.reasoning is None:
            output_cost = usage.completion_tokens / 1_000_000 * self.output
        else:
            normal_output = max(0, usage.completion_tokens - usage.reasoning_tokens)
            output_cost = (
                normal_output / 1_000_000 * self.output
                + usage.reasoning_tokens / 1_000_000 * self.reasoning
            )

        return input_cost + cached_cost + output_cost


@dataclass
class DecisionPolicy:
    """Acceptance and escalation policy for probabilistic decisions."""
    min_probability: Optional[float] = None
    min_confidence: Optional[float] = None
    on_uncertain: Literal["abstain", "return"] = "abstain"
    escalate: bool = False
    escalation_samples: Tuple[int, ...] = (5, 9)

    def validate(self) -> "DecisionPolicy":
        if self.min_probability is not None and not 0.0 <= self.min_probability <= 1.0:
            raise ValueError("min_probability must be between 0 and 1")
        if self.min_confidence is not None and not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.on_uncertain not in ("abstain", "return"):
            raise ValueError("on_uncertain must be 'abstain' or 'return'")
        if any(int(x) < 1 for x in self.escalation_samples):
            raise ValueError("all escalation_samples must be >= 1")
        return self


@dataclass
class ChatResult:
    text: str
    model: Optional[str]
    usage: Usage
    cost: Optional[float]
    raw: Any = field(default=None, repr=False)


@dataclass
class BooleanResult:
    value: bool
    probability: float
    confidence: float
    yes_probability: float
    no_probability: float
    samples: int
    model: Optional[str]
    usage: Usage
    cost: Optional[float]
    explanation: Optional[str] = None
    failed_checks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    raw_samples: List[Dict[str, Any]] = field(default_factory=list, repr=False)


@dataclass
class ChoiceResult:
    value: str
    probability: float
    probabilities: Dict[str, float]
    confidence: float
    samples: int
    model: Optional[str]
    usage: Usage
    cost: Optional[float]
    explanation: Optional[str] = None
    failed_checks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    raw_samples: List[Dict[str, Any]] = field(default_factory=list, repr=False)


@dataclass
class ScoreResult:
    value: float
    distribution: Dict[str, float]
    confidence: float
    samples: int
    model: Optional[str]
    usage: Usage
    cost: Optional[float]
    explanation: Optional[str] = None
    failed_checks: Dict[str, Dict[str, float]] = field(default_factory=dict)
    raw_samples: List[Dict[str, Any]] = field(default_factory=list, repr=False)


@dataclass
class AbstainResult:
    """
    Returned when a decision does not satisfy the requested confidence threshold.

    `candidate` preserves the underlying best-effort decision for inspection.
    """
    abstained: bool
    reason: str
    confidence: float
    threshold: Optional[float]
    candidate: Any
    model: Optional[str]
    usage: Usage
    cost: Optional[float]
    probability: Optional[float] = None
    probability_threshold: Optional[float] = None
    failed_checks: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @property
    def value(self) -> None:
        return None


@dataclass
class AutoResult:
    """Container returned by `auto()` with the inferred decision kind."""

    kind: Literal["boolean", "choice", "score"]
    result: Union[BooleanResult, ChoiceResult, ScoreResult, AbstainResult]
    inferred_spec: Dict[str, Any]

    @property
    def value(self) -> Any:
        return self.result.value

    @property
    def confidence(self) -> float:
        return self.result.confidence

    @property
    def usage(self) -> Usage:
        return self.result.usage

    @property
    def cost(self) -> Optional[float]:
        return self.result.cost

    @property
    def failed_checks(self) -> Dict[str, Dict[str, float]]:
        return getattr(self.result, "failed_checks", {})


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _normalize(weights: Mapping[str, float]) -> Dict[str, float]:
    cleaned = {str(k): max(0.0, float(v)) for k, v in weights.items()}
    total = sum(cleaned.values())
    if total <= 0:
        if not cleaned:
            return {}
        p = 1.0 / len(cleaned)
        return {k: p for k in cleaned}
    return {k: v / total for k, v in cleaned.items()}


def _normalized_entropy(probabilities: Iterable[float]) -> float:
    probs = [p for p in probabilities if p > 0]
    if len(probs) <= 1:
        return 0.0
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(len(probs))


def _distribution_confidence(dist: Mapping[str, float]) -> float:
    """0 = diffuse/uncertain, 1 = concentrated/certain."""
    if not dist:
        return 0.0
    norm = _normalize(dist)
    return _clamp01(1.0 - _normalized_entropy(norm.values()))


def _extract_json(text: str) -> Dict[str, Any]:
    """
    Parse JSON from a model response.

    Supports plain JSON and fenced ```json blocks. As a last resort it scans for
    the first balanced JSON object.
    """
    if not isinstance(text, str):
        raise HmmParseError("Model response content is not text.")

    stripped = text.strip()
    if not stripped:
        raise HmmParseError("Model returned an empty response.")

    candidates = [stripped]

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.I | re.S)
    if fence:
        candidates.insert(0, fence.group(1))

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

    # Balanced-object scan, respecting strings and escapes.
    start = stripped.find("{")
    if start >= 0:
        depth = 0
        in_string = False
        escaped = False
        for i in range(start, len(stripped)):
            ch = stripped[i]
            if in_string:
                if escaped:
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = stripped[start : i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except json.JSONDecodeError:
                        break

    raise HmmParseError(f"Could not parse JSON from model response: {stripped[:300]!r}")


def _usage_from_response(response: Any) -> Usage:
    u = getattr(response, "usage", None)
    if u is None:
        return Usage(requests=1)

    prompt = int(getattr(u, "prompt_tokens", 0) or 0)
    completion = int(getattr(u, "completion_tokens", 0) or 0)
    total = int(getattr(u, "total_tokens", prompt + completion) or 0)

    cached = 0
    prompt_details = getattr(u, "prompt_tokens_details", None)
    if prompt_details is not None:
        cached = int(getattr(prompt_details, "cached_tokens", 0) or 0)

    reasoning = 0
    completion_details = getattr(u, "completion_tokens_details", None)
    if completion_details is not None:
        reasoning = int(getattr(completion_details, "reasoning_tokens", 0) or 0)

    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cached_tokens=cached,
        reasoning_tokens=reasoning,
        requests=1,
    )


class HmmSession:
    """
    Stateful HmmPy session for OpenAI-compatible endpoints.

    Developer:
        Pouriya Khalilian

    Parameters
    ----------
    base_url:
        Base URL of any OpenAI-compatible API, for example
        "https://api.example.com/v1" or "http://localhost:1234/v1".
    api_key:
        Provider key. For local servers that ignore authentication, any placeholder
        string can be used if the client requires one.
    model:
        Provider-specific model identifier.
    system_prompt:
        Optional user-defined system prompt. HmmPy appends its own temporary
        decision instructions only for decision calls.
    timeout:
        Request timeout passed to the OpenAI Python SDK. Defaults to 300 seconds
        to better support slower proxy and local-model providers.
    max_retries:
        Automatic retry count used by the OpenAI Python SDK. Defaults to 2.
    extra_headers:
        Optional provider-specific HTTP headers.
    model_kwargs:
        Default arguments passed to every Chat Completions request, e.g.
        {"temperature": 0.2, "max_tokens": 1000}.
    pricing:
        Optional Pricing object or dict with per-million-token rates.
    keep_decisions_in_history:
        If True, decision prompts/results are stored in the conversational
        history. Default False to avoid polluting ordinary chat state.
    decision history behavior:
        Decision calls ignore chat history by default to reduce token usage and
        context contamination. Pass use_history=True per call, or configure
        decision_history=True for conversational decisions.

    Notes
    -----
    HmmPy probabilities are derived from model-provided distributions and/or
    repeated sampling consensus. They are useful uncertainty signals, but they
    are not guaranteed to be statistically calibrated.

    Example
    -------
    >>> hmm = HmmSession(
    ...     base_url="http://localhost:1234/v1",
    ...     api_key="local",
    ...     model="qwen"
    ... )
    >>> hmm.set_system_prompt("Be concise.")
    >>> r = hmm.boolean("Is this proposal internally consistent?", samples=3)
    >>> print(r.value, r.probability)
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        *,
        system_prompt: Optional[str] = None,
        timeout: float = 300.0,
        max_retries: int = 2,
        extra_headers: Optional[Dict[str, str]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        pricing: Optional[Union[Pricing, Dict[str, Any]]] = None,
        keep_decisions_in_history: bool = False,
        default_policy: Optional[DecisionPolicy] = None,
    ):
        if not base_url:
            raise HmmConfigurationError("base_url is required.")

        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "hmmpy-local"
        self.model = model
        self.system_prompt = system_prompt
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.extra_headers = {
            # Avoid Brotli decoder incompatibilities seen in some Anaconda /
            # OpenAI SDK / httpx2 combinations. Providers can still use gzip
            # or deflate, both of which are widely supported.
            "Accept-Encoding": "gzip, deflate",
            **dict(extra_headers or {}),
        }
        self.model_kwargs: Dict[str, Any] = dict(model_kwargs or {})
        self.keep_decisions_in_history = bool(keep_decisions_in_history)
        self.decision_history = False
        self.default_policy = (default_policy or DecisionPolicy()).validate()

        self._client = self._make_client()
        self._messages: List[Dict[str, str]] = []
        self._total_usage = Usage()
        self._last_usage = Usage()
        self._last_cost: Optional[float] = None
        self._last_result: Any = None

        if pricing is None:
            self.pricing: Optional[Pricing] = None
        elif isinstance(pricing, Pricing):
            self.pricing = pricing
        else:
            self.pricing = Pricing(**pricing)

    # ---------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------

    def _make_client(self) -> OpenAI:
        kwargs: Dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.base_url,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }
        if self.extra_headers:
            kwargs["default_headers"] = self.extra_headers
        return OpenAI(**kwargs)

    def set_system_prompt(self, prompt: Optional[str]) -> "HmmSession":
        self.system_prompt = prompt
        return self

    def set_model(self, model: str, **model_kwargs: Any) -> "HmmSession":
        self.model = model
        if model_kwargs:
            self.model_kwargs.update(model_kwargs)
        return self

    def set_provider(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        extra_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
    ) -> "HmmSession":
        self.base_url = base_url.rstrip("/")
        if api_key is not None:
            self.api_key = api_key
        if extra_headers is not None:
            self.extra_headers = {
                "Accept-Encoding": "gzip, deflate",
                **dict(extra_headers),
            }
        if timeout is not None:
            self.timeout = float(timeout)
        if max_retries is not None:
            self.max_retries = int(max_retries)
        self._client = self._make_client()
        return self

    def set_timeout(self, seconds: float) -> "HmmSession":
        """Set the default request timeout in seconds and rebuild the client."""
        if seconds <= 0:
            raise ValueError("timeout must be > 0")
        self.timeout = float(seconds)
        self._client = self._make_client()
        return self

    def set_retries(self, max_retries: int) -> "HmmSession":
        """Set automatic retry count used by the OpenAI-compatible client."""
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.max_retries = int(max_retries)
        self._client = self._make_client()
        return self

    def set_pricing(
        self,
        *,
        input_per_million: float,
        output_per_million: float,
        cached_input_per_million: Optional[float] = None,
        reasoning_per_million: Optional[float] = None,
        currency: str = "USD",
    ) -> "HmmSession":
        self.pricing = Pricing(
            input=input_per_million,
            output=output_per_million,
            cached_input=cached_input_per_million,
            reasoning=reasoning_per_million,
            currency=currency,
        )
        return self

    def configure(self, **options: Any) -> "HmmSession":
        allowed = {
            "system_prompt",
            "keep_decisions_in_history",
            "model_kwargs",
            "decision_history",
        }
        unknown = set(options) - allowed
        if unknown:
            raise HmmConfigurationError(
                f"Unknown configure option(s): {', '.join(sorted(unknown))}"
            )
        if "system_prompt" in options:
            self.system_prompt = options["system_prompt"]
        if "keep_decisions_in_history" in options:
            self.keep_decisions_in_history = bool(options["keep_decisions_in_history"])
        if "model_kwargs" in options:
            self.model_kwargs.update(dict(options["model_kwargs"]))
        if "decision_history" in options:
            self.decision_history = bool(options["decision_history"])
        return self

    # ---------------------------------------------------------------------
    # Low-level request helpers
    # ---------------------------------------------------------------------

    def _require_model(self) -> None:
        if not self.model:
            raise HmmConfigurationError(
                "No model is configured. Pass model=... or call set_model(...)."
            )

    def _base_messages(self) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(copy.deepcopy(self._messages))
        return messages

    def _request(
        self,
        messages: List[Dict[str, str]],
        *,
        request_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Any, Usage]:
        self._require_model()

        kwargs = dict(self.model_kwargs)
        kwargs.update(request_kwargs or {})

        # These are set explicitly by HmmPy.
        kwargs.pop("model", None)
        kwargs.pop("messages", None)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs,
        )

        try:
            content = response.choices[0].message.content
        except Exception as exc:
            raise HmmPyError("Provider returned an unexpected Chat Completions response.") from exc

        if content is None:
            content = ""

        usage = _usage_from_response(response)
        self._record_usage(usage)
        return str(content), response, usage

    def _record_usage(self, usage: Usage) -> None:
        self._last_usage = usage
        self._total_usage = self._total_usage + usage
        self._last_cost = self._estimate_cost(usage)

    def _estimate_cost(self, usage: Usage) -> Optional[float]:
        if self.pricing is None:
            return None
        return self.pricing.estimate(usage)

    def _aggregate_call_usage(self, parts: Sequence[Usage]) -> Usage:
        total = Usage()
        for part in parts:
            total = total + part
        return total

    def _decision_request(
        self,
        prompt: str,
        instruction: str,
        *,
        request_kwargs: Optional[Dict[str, Any]] = None,
        use_history: bool = True,
    ) -> Tuple[Dict[str, Any], Any, Usage, str]:
        if use_history:
            messages = self._base_messages()
        else:
            messages = []
            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
        messages.append(
            {
                "role": "system",
                "content": (
                    "HmmPy decision protocol: Return ONLY one valid JSON object. "
                    "Do not use markdown fences. Do not add text before or after JSON. "
                    "Treat probabilities as your uncertainty estimates, not guarantees. "
                    + instruction
                ),
            }
        )
        messages.append({"role": "user", "content": prompt})

        content, raw, usage = self._request(messages, request_kwargs=request_kwargs)
        return _extract_json(content), raw, usage, content

    # ---------------------------------------------------------------------
    # Conversation
    # ---------------------------------------------------------------------

    def chat(self, prompt: str, **request_kwargs: Any) -> ChatResult:
        """Send a normal conversational message and keep it in session history."""
        messages = self._base_messages()
        messages.append({"role": "user", "content": prompt})
        content, raw, usage = self._request(messages, request_kwargs=request_kwargs)

        self._messages.append({"role": "user", "content": prompt})
        self._messages.append({"role": "assistant", "content": content})

        result = ChatResult(
            text=content,
            model=getattr(raw, "model", self.model),
            usage=usage,
            cost=self._estimate_cost(usage),
            raw=raw,
        )
        self._last_result = result
        return result

    ask = chat

    def set_policy(self, policy: DecisionPolicy) -> "HmmSession":
        self.default_policy = policy.validate()
        return self

    def _resolve_policy(self, *, policy=None, min_probability=None, min_confidence=None, abstain=None, escalate=None) -> DecisionPolicy:
        p = copy.deepcopy(policy or self.default_policy)
        if min_probability is not None: p.min_probability = float(min_probability)
        if min_confidence is not None: p.min_confidence = float(min_confidence)
        if abstain is not None: p.on_uncertain = "abstain" if abstain else "return"
        if escalate is not None: p.escalate = bool(escalate)
        return p.validate()

    def _decision_probability(self, result: Any) -> Optional[float]:
        if isinstance(result, (BooleanResult, ChoiceResult)):
            return float(result.probability)
        if isinstance(result, ScoreResult) and result.distribution:
            return float(max(result.distribution.values()))
        return None

    def _policy_failed_checks(
        self,
        result: Any,
        policy: DecisionPolicy,
    ) -> Dict[str, Dict[str, float]]:
        """Return all policy thresholds that the candidate decision failed."""
        failed: Dict[str, Dict[str, float]] = {}

        probability = self._decision_probability(result)
        if policy.min_probability is not None:
            probability_value = float(probability) if probability is not None else 0.0
            if probability is None or probability_value < policy.min_probability:
                failed["probability"] = {
                    "value": probability_value,
                    "required": float(policy.min_probability),
                }

        confidence = float(getattr(result, "confidence", 0.0))
        if policy.min_confidence is not None and confidence < policy.min_confidence:
            failed["confidence"] = {
                "value": confidence,
                "required": float(policy.min_confidence),
            }

        return failed

    def _policy_failure_reason(
        self,
        result: Any,
        policy: DecisionPolicy,
    ) -> Optional[str]:
        """Return a short backward-compatible summary of policy failure."""
        failed = self._policy_failed_checks(result, policy)
        if "probability" in failed:
            return "probability below threshold"
        if "confidence" in failed:
            return "confidence below threshold"
        return None

    def _apply_policy(self, result: Any, policy: DecisionPolicy) -> Any:
        failed_checks = self._policy_failed_checks(result, policy)
        reason = self._policy_failure_reason(result, policy)
        if hasattr(result, "failed_checks"):
            result.failed_checks = failed_checks

        if reason is None or policy.on_uncertain == "return":
            return result
        out = AbstainResult(
            abstained=True, reason=reason, confidence=float(getattr(result, "confidence", 0.0)),
            threshold=policy.min_confidence, candidate=result, model=getattr(result, "model", self.model),
            usage=getattr(result, "usage", Usage()), cost=getattr(result, "cost", None),
            probability=self._decision_probability(result), probability_threshold=policy.min_probability,
            failed_checks=failed_checks,
        )
        self._last_result = out
        return out

    # ---------------------------------------------------------------------
    # Jev-like decision primitives
    # ---------------------------------------------------------------------

    def boolean(
        self,
        prompt: str,
        *,
        samples: int = 3,
        include_explanation: bool = True,
        min_probability: Optional[float] = None,
        min_confidence: Optional[float] = None,
        policy: Optional[DecisionPolicy] = None,
        abstain: Optional[bool] = None,
        escalate: Optional[bool] = None,
        use_history: Optional[bool] = None,
        **request_kwargs: Any,
    ) -> Union[BooleanResult, AbstainResult]:
        """
        Make a Yes/No decision.

        Each sample returns a boolean plus a model-estimated probability that the
        answer is True. HmmPy averages those probabilities across samples.
        """
        if use_history is None:
            use_history = self.decision_history
        if samples < 1:
            raise ValueError("samples must be >= 1")

        instruction = (
            'Schema: {"value": true|false, "yes_probability": number 0..1'
            + (', "explanation": "short explanation"' if include_explanation else "")
            + "}. `yes_probability` means P(True)."
        )

        parsed_samples: List[Dict[str, Any]] = []
        usages: List[Usage] = []
        raw_responses: List[Any] = []

        for _ in range(samples):
            obj, raw, usage, _ = self._decision_request(
                prompt, instruction, request_kwargs=request_kwargs, use_history=use_history
            )
            yes_p = _clamp01(float(obj.get("yes_probability", 1.0 if obj.get("value") else 0.0)))
            value = bool(obj.get("value", yes_p >= 0.5))
            parsed_samples.append(
                {
                    "value": value,
                    "yes_probability": yes_p,
                    "explanation": obj.get("explanation"),
                }
            )
            usages.append(usage)
            raw_responses.append(raw)

        yes = statistics.fmean(x["yes_probability"] for x in parsed_samples)
        no = 1.0 - yes
        value = yes >= 0.5
        probability = yes if value else no

        # Boolean confidence is distance from a maximally uncertain 0.5 prediction.
        confidence = _clamp01(abs(yes - 0.5) * 2.0)

        explanation = None
        if include_explanation:
            matching = [x["explanation"] for x in parsed_samples if x["value"] == value and x["explanation"]]
            if matching:
                explanation = matching[0]

        total_usage = self._aggregate_call_usage(usages)
        model = getattr(raw_responses[-1], "model", self.model) if raw_responses else self.model
        result = BooleanResult(
            value=value,
            probability=probability,
            confidence=confidence,
            yes_probability=yes,
            no_probability=no,
            samples=samples,
            model=model,
            usage=total_usage,
            cost=self._estimate_cost(total_usage),
            explanation=explanation,
            raw_samples=parsed_samples,
        )
        self._last_result = result
        self._last_usage = total_usage
        self._last_cost = self._estimate_cost(total_usage)

        if self.keep_decisions_in_history:
            self._messages.append({"role": "user", "content": prompt})
            self._messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"value": result.value, "probability": result.probability},
                        ensure_ascii=False,
                    ),
                }
            )
        active_policy = self._resolve_policy(policy=policy, min_probability=min_probability, min_confidence=min_confidence, abstain=abstain, escalate=escalate)
        escalation_usage = result.usage
        if self._policy_failure_reason(result, active_policy) and active_policy.escalate:
            for n in active_policy.escalation_samples:
                n = int(n)
                if n <= samples:
                    continue
                candidate = self.boolean(
                    prompt,
                    samples=n,
                    include_explanation=include_explanation,
                    use_history=use_history,
                    policy=DecisionPolicy(on_uncertain="return", escalate=False),
                    abstain=False,
                    escalate=False,
                    **request_kwargs,
                )
                escalation_usage = escalation_usage + candidate.usage
                result = candidate
                result.usage = escalation_usage
                result.cost = self._estimate_cost(escalation_usage)
                self._last_usage = escalation_usage
                self._last_cost = result.cost
                if self._policy_failure_reason(result, active_policy) is None:
                    break
        return self._apply_policy(result, active_policy)

    def choice(
        self,
        prompt: str,
        choices: Sequence[str],
        *,
        samples: int = 3,
        include_explanation: bool = True,
        min_probability: Optional[float] = None,
        min_confidence: Optional[float] = None,
        policy: Optional[DecisionPolicy] = None,
        abstain: Optional[bool] = None,
        escalate: Optional[bool] = None,
        use_history: Optional[bool] = None,
        **request_kwargs: Any,
    ) -> Union[ChoiceResult, AbstainResult]:
        """Choose one item and return an aggregated probability distribution."""
        if use_history is None:
            use_history = self.decision_history
        choices = [str(x) for x in choices]
        if len(choices) < 2:
            raise ValueError("choice() requires at least 2 choices.")
        if len(set(choices)) != len(choices):
            raise ValueError("choices must be unique.")
        if samples < 1:
            raise ValueError("samples must be >= 1")

        instruction = (
            "Allowed choices: "
            + json.dumps(choices, ensure_ascii=False)
            + '. Schema: {"value": "one exact allowed choice", '
              '"probabilities": {"choice": probability, "...": probability}'
            + (', "explanation": "short explanation"' if include_explanation else "")
            + "}. Include every allowed choice in `probabilities`; probabilities must sum approximately to 1."
        )

        parsed_samples: List[Dict[str, Any]] = []
        usages: List[Usage] = []
        raw_responses: List[Any] = []

        totals = {choice: 0.0 for choice in choices}

        for _ in range(samples):
            obj, raw, usage, _ = self._decision_request(
                prompt, instruction, request_kwargs=request_kwargs, use_history=use_history
            )

            raw_dist = obj.get("probabilities") or {}
            dist = {c: float(raw_dist.get(c, 0.0) or 0.0) for c in choices}

            # If the provider omitted a useful distribution, fall back to the selected value.
            if sum(max(0.0, v) for v in dist.values()) <= 0:
                selected = str(obj.get("value", ""))
                if selected not in choices:
                    raise HmmParseError(
                        f"Model returned invalid choice {selected!r}; expected one of {choices!r}."
                    )
                dist[selected] = 1.0

            dist = _normalize(dist)
            selected = str(obj.get("value", max(dist, key=dist.get)))
            if selected not in choices:
                selected = max(dist, key=dist.get)

            parsed_samples.append(
                {
                    "value": selected,
                    "probabilities": dist,
                    "explanation": obj.get("explanation"),
                }
            )
            for c in choices:
                totals[c] += dist[c]
            usages.append(usage)
            raw_responses.append(raw)

        aggregate = _normalize({c: totals[c] / samples for c in choices})
        value = max(aggregate, key=aggregate.get)
        probability = aggregate[value]
        confidence = _distribution_confidence(aggregate)

        explanation = None
        if include_explanation:
            matching = [x["explanation"] for x in parsed_samples if x["value"] == value and x["explanation"]]
            if matching:
                explanation = matching[0]

        total_usage = self._aggregate_call_usage(usages)
        model = getattr(raw_responses[-1], "model", self.model) if raw_responses else self.model
        result = ChoiceResult(
            value=value,
            probability=probability,
            probabilities=aggregate,
            confidence=confidence,
            samples=samples,
            model=model,
            usage=total_usage,
            cost=self._estimate_cost(total_usage),
            explanation=explanation,
            raw_samples=parsed_samples,
        )
        self._last_result = result
        self._last_usage = total_usage
        self._last_cost = self._estimate_cost(total_usage)

        if self.keep_decisions_in_history:
            self._messages.append({"role": "user", "content": prompt})
            self._messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"value": result.value, "probabilities": result.probabilities},
                        ensure_ascii=False,
                    ),
                }
            )
        active_policy = self._resolve_policy(policy=policy, min_probability=min_probability, min_confidence=min_confidence, abstain=abstain, escalate=escalate)
        escalation_usage = result.usage
        if self._policy_failure_reason(result, active_policy) and active_policy.escalate:
            for n in active_policy.escalation_samples:
                n = int(n)
                if n <= samples:
                    continue
                candidate = self.choice(
                    prompt,
                    choices=choices,
                    samples=n,
                    include_explanation=include_explanation,
                    use_history=use_history,
                    policy=DecisionPolicy(on_uncertain="return", escalate=False),
                    abstain=False,
                    escalate=False,
                    **request_kwargs,
                )
                escalation_usage = escalation_usage + candidate.usage
                result = candidate
                result.usage = escalation_usage
                result.cost = self._estimate_cost(escalation_usage)
                self._last_usage = escalation_usage
                self._last_cost = result.cost
                if self._policy_failure_reason(result, active_policy) is None:
                    break
        return self._apply_policy(result, active_policy)

    def score(
        self,
        prompt: str,
        scale: Sequence[Union[str, int, float]],
        *,
        samples: int = 3,
        include_explanation: bool = True,
        min_probability: Optional[float] = None,
        min_confidence: Optional[float] = None,
        policy: Optional[DecisionPolicy] = None,
        abstain: Optional[bool] = None,
        escalate: Optional[bool] = None,
        use_history: Optional[bool] = None,
        **request_kwargs: Any,
    ) -> Union[ScoreResult, AbstainResult]:
        """
        Score on an ordered discrete scale.

        The returned `value` is the expected position/value. Numeric scales return
        the expected numeric score. Non-numeric scales return the expected 1-based
        ordinal position.
        """
        if use_history is None:
            use_history = self.decision_history
        values = list(scale)
        if len(values) < 2:
            raise ValueError("score() requires at least 2 scale values.")
        labels = [str(x) for x in values]
        if len(set(labels)) != len(labels):
            raise ValueError("scale values must be unique after string conversion.")
        if samples < 1:
            raise ValueError("samples must be >= 1")

        instruction = (
            "Ordered scale: "
            + json.dumps(labels, ensure_ascii=False)
            + '. Schema: {"value": "one exact scale label", '
              '"distribution": {"label": probability, "...": probability}'
            + (', "explanation": "short explanation"' if include_explanation else "")
            + "}. Include every scale label in `distribution`; probabilities must sum approximately to 1."
        )

        parsed_samples: List[Dict[str, Any]] = []
        usages: List[Usage] = []
        raw_responses: List[Any] = []
        totals = {label: 0.0 for label in labels}

        for _ in range(samples):
            obj, raw, usage, _ = self._decision_request(
                prompt, instruction, request_kwargs=request_kwargs, use_history=use_history
            )
            raw_dist = obj.get("distribution") or obj.get("probabilities") or {}
            dist = {label: float(raw_dist.get(label, 0.0) or 0.0) for label in labels}

            if sum(max(0.0, v) for v in dist.values()) <= 0:
                selected = str(obj.get("value", ""))
                if selected not in labels:
                    raise HmmParseError(
                        f"Model returned invalid score {selected!r}; expected one of {labels!r}."
                    )
                dist[selected] = 1.0

            dist = _normalize(dist)
            selected = str(obj.get("value", max(dist, key=dist.get)))
            if selected not in labels:
                selected = max(dist, key=dist.get)

            parsed_samples.append(
                {
                    "value": selected,
                    "distribution": dist,
                    "explanation": obj.get("explanation"),
                }
            )
            for label in labels:
                totals[label] += dist[label]
            usages.append(usage)
            raw_responses.append(raw)

        aggregate = _normalize({label: totals[label] / samples for label in labels})
        confidence = _distribution_confidence(aggregate)

        numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        if numeric:
            value = sum(float(v) * aggregate[str(v)] for v in values)
        else:
            value = sum((i + 1) * aggregate[label] for i, label in enumerate(labels))

        explanation = None
        if include_explanation:
            mode = max(aggregate, key=aggregate.get)
            matching = [x["explanation"] for x in parsed_samples if x["value"] == mode and x["explanation"]]
            if matching:
                explanation = matching[0]

        total_usage = self._aggregate_call_usage(usages)
        model = getattr(raw_responses[-1], "model", self.model) if raw_responses else self.model
        result = ScoreResult(
            value=float(value),
            distribution=aggregate,
            confidence=confidence,
            samples=samples,
            model=model,
            usage=total_usage,
            cost=self._estimate_cost(total_usage),
            explanation=explanation,
            raw_samples=parsed_samples,
        )
        self._last_result = result
        self._last_usage = total_usage
        self._last_cost = self._estimate_cost(total_usage)

        if self.keep_decisions_in_history:
            self._messages.append({"role": "user", "content": prompt})
            self._messages.append(
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {"value": result.value, "distribution": result.distribution},
                        ensure_ascii=False,
                    ),
                }
            )
        active_policy = self._resolve_policy(policy=policy, min_probability=min_probability, min_confidence=min_confidence, abstain=abstain, escalate=escalate)
        escalation_usage = result.usage
        if self._policy_failure_reason(result, active_policy) and active_policy.escalate:
            for n in active_policy.escalation_samples:
                n = int(n)
                if n <= samples:
                    continue
                candidate = self.score(
                    prompt,
                    scale=scale,
                    samples=n,
                    include_explanation=include_explanation,
                    use_history=use_history,
                    policy=DecisionPolicy(on_uncertain="return", escalate=False),
                    abstain=False,
                    escalate=False,
                    **request_kwargs,
                )
                escalation_usage = escalation_usage + candidate.usage
                result = candidate
                result.usage = escalation_usage
                result.cost = self._estimate_cost(escalation_usage)
                self._last_usage = escalation_usage
                self._last_cost = result.cost
                if self._policy_failure_reason(result, active_policy) is None:
                    break
        return self._apply_policy(result, active_policy)

    def auto(
        self,
        prompt: str,
        *,
        samples: int = 3,
        min_probability: Optional[float] = None,
        min_confidence: Optional[float] = None,
        policy: Optional[DecisionPolicy] = None,
        abstain: Optional[bool] = None,
        escalate: Optional[bool] = None,
        use_history: Optional[bool] = None,
        **request_kwargs: Any,
    ) -> AutoResult:
        """
        Infer boolean/choice/score from the prompt, then execute that decision.

        For ambiguous tasks, explicit boolean(), choice(), or score() is preferred.
        """
        if use_history is None:
            use_history = self.decision_history
        infer_instruction = (
            'Infer the most appropriate decision type. Schema: '
            '{"kind": "boolean"|"choice"|"score", '
            '"choices": ["..."], "scale": ["..."], '
            '"decision_question": "concise normalized decision question"}. '
            "For boolean, choices and scale must be empty arrays. "
            "For choice, return at least 2 explicit choices. "
            "For score, return an ordered scale with at least 2 labels. "
            "Do not invent choice categories if the user has clearly provided them."
        )

        spec, _, infer_usage, _ = self._decision_request(
            prompt, infer_instruction, request_kwargs=request_kwargs, use_history=use_history
        )
        kind = str(spec.get("kind", "")).lower()
        question = str(spec.get("decision_question") or prompt)

        # The inference request already incremented session totals. The delegated
        # decision call will add its own usage; AutoResult reports both.
        if kind == "boolean":
            result = self.boolean(
                question,
                samples=samples,
                min_probability=None,
                min_confidence=None,
                use_history=use_history,
                policy=DecisionPolicy(on_uncertain="return", escalate=False),
                abstain=False,
                escalate=False,
                **request_kwargs,
            )
        elif kind == "choice":
            choices = spec.get("choices") or []
            if len(choices) < 2:
                raise HmmParseError("auto() inferred choice but did not return >=2 choices.")
            result = self.choice(
                question,
                choices=choices,
                samples=samples,
                min_probability=None,
                min_confidence=None,
                use_history=use_history,
                policy=DecisionPolicy(on_uncertain="return", escalate=False),
                abstain=False,
                escalate=False,
                **request_kwargs,
            )
        elif kind == "score":
            scale = spec.get("scale") or []
            if len(scale) < 2:
                raise HmmParseError("auto() inferred score but did not return >=2 scale values.")
            result = self.score(
                question,
                scale=scale,
                samples=samples,
                min_probability=None,
                min_confidence=None,
                use_history=use_history,
                policy=DecisionPolicy(on_uncertain="return", escalate=False),
                abstain=False,
                escalate=False,
                **request_kwargs,
            )
        else:
            raise HmmParseError(f"auto() received unknown inferred decision kind: {kind!r}")

        combined_usage = infer_usage + result.usage
        result.usage = combined_usage
        result.cost = self._estimate_cost(combined_usage)
        self._last_usage = combined_usage
        self._last_cost = result.cost

        active_policy = self._resolve_policy(policy=policy, min_probability=min_probability, min_confidence=min_confidence, abstain=abstain, escalate=escalate)
        auto_escalation_usage = result.usage
        if self._policy_failure_reason(result, active_policy) and active_policy.escalate:
            for n in active_policy.escalation_samples:
                n = int(n)
                if n <= samples:
                    continue
                if kind == "boolean":
                    candidate = self.boolean(question, samples=n, use_history=use_history, policy=DecisionPolicy(on_uncertain="return"), abstain=False, escalate=False, **request_kwargs)
                elif kind == "choice":
                    candidate = self.choice(question, choices=spec.get("choices") or [], samples=n, use_history=use_history, policy=DecisionPolicy(on_uncertain="return"), abstain=False, escalate=False, **request_kwargs)
                else:
                    candidate = self.score(question, scale=spec.get("scale") or [], samples=n, use_history=use_history, policy=DecisionPolicy(on_uncertain="return"), abstain=False, escalate=False, **request_kwargs)
                auto_escalation_usage = auto_escalation_usage + candidate.usage
                result = candidate
                result.usage = auto_escalation_usage
                result.cost = self._estimate_cost(auto_escalation_usage)
                self._last_usage = auto_escalation_usage
                self._last_cost = result.cost
                if self._policy_failure_reason(result, active_policy) is None:
                    break
        final_result = self._apply_policy(result, active_policy)
        auto_result = AutoResult(kind=kind, result=final_result, inferred_spec=spec)
        self._last_result = auto_result
        return auto_result

    decide = auto

    # ---------------------------------------------------------------------
    # History/state
    # ---------------------------------------------------------------------

    def messages(self) -> List[Dict[str, str]]:
        return copy.deepcopy(self._messages)

    def history(self) -> List[Dict[str, str]]:
        return self.messages()

    def clear(self, *, reset_usage: bool = False) -> "HmmSession":
        self._messages.clear()
        self._last_result = None
        if reset_usage:
            self._total_usage = Usage()
            self._last_usage = Usage()
            self._last_cost = None
        return self

    def reset(self) -> "HmmSession":
        return self.clear(reset_usage=True)

    def fork(self) -> "HmmSession":
        cloned = HmmSession(
            base_url=self.base_url,
            api_key=self.api_key,
            model=self.model,
            system_prompt=self.system_prompt,
            timeout=self.timeout,
            max_retries=self.max_retries,
            extra_headers=self.extra_headers,
            model_kwargs=copy.deepcopy(self.model_kwargs),
            pricing=copy.deepcopy(self.pricing),
            keep_decisions_in_history=self.keep_decisions_in_history,
            default_policy=copy.deepcopy(self.default_policy),
        )
        cloned._messages = copy.deepcopy(self._messages)
        cloned.decision_history = self.decision_history
        return cloned

    def is_abstained(self, result: Any = None) -> bool:
        """Return True if `result` (or the last result) is an abstention."""
        target = self._last_result if result is None else result
        if isinstance(target, AutoResult):
            target = target.result
        return isinstance(target, AbstainResult)

    # ---------------------------------------------------------------------
    # Usage/cost/inspection
    # ---------------------------------------------------------------------

    def usage(self) -> Usage:
        return copy.deepcopy(self._total_usage)

    def last_usage(self) -> Usage:
        return copy.deepcopy(self._last_usage)

    def cost(self) -> Optional[float]:
        return self._estimate_cost(self._total_usage)

    def last_cost(self) -> Optional[float]:
        return self._last_cost

    @property
    def tokens(self) -> int:
        return self._total_usage.total_tokens

    @property
    def last_tokens(self) -> int:
        return self._last_usage.total_tokens

    @property
    def total_cost(self) -> Optional[float]:
        return self.cost()

    @property
    def last_response(self) -> Any:
        return self._last_result

    def stats(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "base_url": self.base_url,
            "usage": self.usage().to_dict(),
            "cost": self.cost(),
            "currency": self.pricing.currency if self.pricing else None,
            "messages": len(self._messages),
        }

    def model_info(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "model_kwargs": copy.deepcopy(self.model_kwargs),
            "pricing": asdict(self.pricing) if self.pricing else None,
        }

    def provider_info(self) -> Dict[str, Any]:
        # Never expose the API key.
        return {
            "base_url": self.base_url,
            "extra_headers": copy.deepcopy(self.extra_headers),
            "timeout": self.timeout,
            "max_retries": self.max_retries,
        }

    def config(self) -> Dict[str, Any]:
        return {
            **self.provider_info(),
            **self.model_info(),
            "system_prompt": self.system_prompt,
            "keep_decisions_in_history": self.keep_decisions_in_history,
            "decision_history": self.decision_history,
            "default_policy": asdict(self.default_policy),
        }

    def last_result(self) -> Any:
        return self._last_result


def about() -> str:
    """Return package metadata suitable for notebooks and `print(about())`."""
    return (
        f"{__title__} {__version__}\n"
        "Probabilistic decisions for OpenAI-compatible language models.\n\n"
        f"Developer: {__author__}\n"
        f"License: {__license__}"
    )


__all__ = [
    "HmmSession",
    "DecisionPolicy",
    "Usage",
    "Pricing",
    "ChatResult",
    "BooleanResult",
    "ChoiceResult",
    "ScoreResult",
    "AutoResult",
    "AbstainResult",
    "HmmPyError",
    "HmmParseError",
    "HmmConfigurationError",
    "about",
]

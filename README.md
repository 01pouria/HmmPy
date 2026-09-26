# HmmPy

**Typed probabilistic decisions for OpenAI-compatible language models.**

Developer: **Pouriya Khalilian**  
Status: **Alpha / v0.3.3**

HmmPy is a lightweight Python decision layer that works with any OpenAI-compatible endpoint. It turns natural-language tasks into typed probabilistic decisions while tracking uncertainty, token usage, and optional cost estimates.

## Features

- OpenAI-compatible cloud or local providers
- Stateful `HmmSession`
- `boolean()`, `choice()`, `score()`, and `auto()`
- Multi-sample aggregation
- Probability and confidence signals
- `DecisionPolicy`
- Probability/confidence thresholds
- Abstention and sample escalation
- Optional conversation history
- Token usage tracking
- Cost estimation with user-supplied pricing
- No DSPy dependency required

## Installation

```bash
pip install hmmpy
```

For local development:

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
from hmmpy import HmmSession

hmm = HmmSession(
    base_url="https://your-provider.example/v1",
    api_key="YOUR_API_KEY",
    model="YOUR_MODEL",
)

result = hmm.boolean(
    "Is Python a programming language?",
    samples=3,
    min_probability=0.90,
    min_confidence=0.80,
)

print(result.value)
print(result.probability)
print(result.confidence)
print(result.failed_checks)
print(result.usage)
```

## Local models

Any OpenAI-compatible Chat Completions endpoint can be used:

```python
hmm = HmmSession(
    base_url="http://localhost:1234/v1",
    api_key="local",
    model="my-local-model",
)
```

## Decision primitives

```python
result = hmm.choice(
    "Which category best describes Python?",
    choices=["programming language", "database", "operating system"],
    samples=3,
)
print(result.value)
print(result.probabilities)
```

```python
result = hmm.score(
    "Rate Python's suitability for ML prototyping.",
    scale=[1, 2, 3, 4, 5],
    samples=3,
)
print(result.value)
print(result.distribution)
```

```python
result = hmm.auto("Is this evidence sufficient to accept the hypothesis?", samples=3)
print(result.kind, result.value, result.confidence)
```

## Policy, escalation, and abstention

```python
from hmmpy import DecisionPolicy

policy = DecisionPolicy(
    min_probability=0.90,
    min_confidence=0.80,
    on_uncertain="abstain",
    escalate=True,
    escalation_samples=(5, 9),
)

result = hmm.boolean("Is the evidence sufficient?", samples=3, policy=policy)

if hmm.is_abstained(result):
    print(result.failed_checks)
else:
    print(result.value)
```

## Conversation history

Decision calls ignore chat history by default to reduce token usage and context contamination. Enable it explicitly when the decision depends on the conversation:

```python
hmm.chat("The project codename is Aurora.")
result = hmm.boolean(
    "Based on our earlier conversation, is the codename Aurora?",
    use_history=True,
)
```

## Usage and cost

```python
print(hmm.last_usage())
print(hmm.usage())
print(hmm.tokens)
print(hmm.stats())
```

Pricing is not standardized by the OpenAI-compatible protocol, so configure it manually:

```python
hmm.set_pricing(input_per_million=0.50, output_per_million=2.00)
print(hmm.cost())
```

## Probability note

HmmPy probabilities are **model-derived uncertainty estimates** produced through model-reported distributions and repeated sampling. They are **not guaranteed to be statistically calibrated probabilities**. For production use where calibration matters, evaluate on labeled data and add a calibration layer.

## Security

Never commit API keys. Use environment variables, secret managers, or local secure configuration.

## License

HmmPy is released under the **MIT License**. See `LICENSE` for the full text.

## Development

```bash
pytest
python -m build
python -m twine check dist/*
```

## Roadmap

- model routing: cheap → balanced → strong
- async client support
- calibration utilities
- provider capability probing
- optional DSPy integration
- richer evaluation metrics

## Repository

https://github.com/01pouria/hmmpy

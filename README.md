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
pip install hmmpy-ai
```

The PyPI distribution is named `hmmpy-ai`, while the Python import remains:

```python
import hmmpy
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

## Local Models

Any OpenAI-compatible Chat Completions endpoint can be used:

```python
hmm = HmmSession(
    base_url="http://localhost:1234/v1",
    api_key="local",
    model="my-local-model",
)
```

## Decision Primitives

### Boolean

```python
result = hmm.boolean(
    "Is Python a programming language?",
    samples=3,
)

print(result.value)
print(result.probability)
print(result.confidence)
```

### Choice

```python
result = hmm.choice(
    "Which category best describes Python?",
    choices=[
        "programming language",
        "database",
        "operating system",
    ],
    samples=3,
)

print(result.value)
print(result.probabilities)
```

### Score

```python
result = hmm.score(
    "Rate Python's suitability for ML prototyping.",
    scale=[1, 2, 3, 4, 5],
    samples=3,
)

print(result.value)
print(result.distribution)
```

### Automatic Decision Type

```python
result = hmm.auto(
    "Is this evidence sufficient to accept the hypothesis?",
    samples=3,
)

print(result.kind)
print(result.value)
print(result.confidence)
```

## Policy, Escalation, and Abstention

HmmPy can enforce probability and confidence requirements before accepting a decision.

```python
from hmmpy import DecisionPolicy

policy = DecisionPolicy(
    min_probability=0.90,
    min_confidence=0.80,
    on_uncertain="abstain",
    escalate=True,
    escalation_samples=(5, 9),
)

result = hmm.boolean(
    "Is the evidence sufficient?",
    samples=3,
    policy=policy,
)

if hmm.is_abstained(result):
    print("Abstained")
    print(result.failed_checks)
else:
    print(result.value)
```

When escalation is enabled, HmmPy can increase the number of samples before abstaining.

For example:

```text
3 samples
    ↓ uncertain
5 samples
    ↓ uncertain
9 samples
    ↓
decision or abstain
```

## Conversation History

Decision calls ignore chat history by default to reduce token usage and context contamination.

Enable history explicitly when the decision depends on the conversation:

```python
hmm.chat("The project codename is Aurora.")

result = hmm.boolean(
    "Based on our earlier conversation, is the codename Aurora?",
    use_history=True,
)
```

## Usage and Cost

HmmPy tracks token usage when the OpenAI-compatible provider exposes usage information.

```python
print(hmm.last_usage())
print(hmm.usage())
print(hmm.tokens)
print(hmm.stats())
```

Pricing is not standardized by the OpenAI-compatible protocol, so provider pricing can be configured manually:

```python
hmm.set_pricing(
    input_per_million=0.50,
    output_per_million=2.00,
)

print(hmm.cost())
```

## Probability Note

HmmPy probabilities are **model-derived uncertainty estimates** produced through model-reported distributions and repeated sampling.

They are **not guaranteed to be statistically calibrated probabilities**.

For production systems where calibration matters, evaluate the model on labeled data and apply an appropriate calibration method.

## Security

Never commit API keys to source control.

Use environment variables, secret managers, or local secure configuration.

Example:

```python
import os
from hmmpy import HmmSession

hmm = HmmSession(
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ["OPENAI_MODEL"],
)
```

## Development

Clone the repository:

```bash
git clone https://github.com/01pouria/HmmPy.git
cd HmmPy
```

Create and activate a virtual environment, then install the development dependencies:

```bash
pip install -e ".[dev]"
```

Run the test suite:

```bash
pytest
```

Build the package:

```bash
python -m build
```

Validate the distributions:

```bash
python -m twine check dist/*
```

## Roadmap

- Model routing: cheap → balanced → strong
- Async client support
- Calibration utilities
- Provider capability probing
- Optional DSPy integration
- Richer evaluation metrics

## License

HmmPy is released under the **MIT License**.

See [`LICENSE`](LICENSE) for the full license text.

## Repository

GitHub: https://github.com/01pouria/HmmPy

## Install from PyPI

After the first public release:

```bash
pip install hmmpy-ai
```

Then use it as:

```python
from hmmpy import HmmSession
```
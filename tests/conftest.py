import sys
import types
from types import SimpleNamespace
import pytest

# Stub the OpenAI SDK at import time so pure unit tests do not require network or
# a real provider. The package dependency is still declared in pyproject.toml.
_openai = types.ModuleType("openai")
class _DummyOpenAI:
    def __init__(self, *args, **kwargs):
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=None))
_openai.OpenAI = _DummyOpenAI
sys.modules.setdefault("openai", _openai)

from hmmpy import HmmSession

class FakeCompletions:
    def __init__(self, contents):
        self.contents = list(contents)
        self.calls = []
    def create(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, "kwargs": kwargs})
        if not self.contents:
            raise AssertionError("No fake responses left")
        content = self.contents.pop(0)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=5, total_tokens=15,
                                prompt_tokens_details=None, completion_tokens_details=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               usage=usage, model=model)

class FakeClient:
    def __init__(self, contents):
        self.chat = SimpleNamespace(completions=FakeCompletions(contents))

@pytest.fixture
def make_session():
    def factory(contents):
        hmm = HmmSession(base_url="http://example.test/v1", api_key="test", model="test-model")
        hmm._client = FakeClient(contents)
        return hmm
    return factory

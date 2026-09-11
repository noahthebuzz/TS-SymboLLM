from ts_symbollm.backends.base import Backend
from ts_symbollm.backends.ollama import OllamaBackend


class _DummyBackend(Backend):
    def generate(self, model, prompt, params=None):
        return "dummy response"


def test_dummy_backend_satisfies_the_interface():
    backend = _DummyBackend()
    assert backend.generate("model", "prompt") == "dummy response"


def test_backend_pull_defaults_to_a_noop_true():
    backend = _DummyBackend()
    assert backend.pull("any-model") is True


def test_ollama_backend_is_a_backend():
    assert isinstance(OllamaBackend(), Backend)


def test_ollama_backend_generate_streams_and_returns_full_text(monkeypatch):
    def fake_generate(model, prompt, stream, options=None):
        assert model == "qwen2.5:7b"
        assert prompt == "hello"
        return iter([{"response": "foo"}, {"response": "bar"}])

    monkeypatch.setattr("ts_symbollm.backends.ollama.ollama.generate", fake_generate)

    backend = OllamaBackend()
    result = backend.generate(model="qwen2.5:7b", prompt="hello")
    assert result == "foobar"


def test_ollama_backend_generate_passes_params_through(monkeypatch):
    captured = {}

    def fake_generate(model, prompt, stream, options=None):
        captured["options"] = options
        return iter([{"response": "ok"}])

    monkeypatch.setattr("ts_symbollm.backends.ollama.ollama.generate", fake_generate)

    backend = OllamaBackend()
    backend.generate(model="m", prompt="p", params={"temperature": 0.1})
    assert captured["options"] == {"temperature": 0.1}


def test_ollama_backend_pull_reports_success(monkeypatch):
    monkeypatch.setattr(
        "ts_symbollm.backends.ollama.ollama.pull",
        lambda model, stream: iter([{"status": "success"}]),
    )
    backend = OllamaBackend()
    assert backend.pull("qwen2.5:7b") is True


def test_ollama_backend_pull_reports_failure_on_exception(monkeypatch):
    def raise_error(model, stream):
        raise RuntimeError("boom")

    monkeypatch.setattr("ts_symbollm.backends.ollama.ollama.pull", raise_error)
    backend = OllamaBackend()
    assert backend.pull("qwen2.5:7b") is False

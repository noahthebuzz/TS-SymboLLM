from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class Backend(ABC):
    '''
    Minimal interface a model-serving backend must implement, so the rest
    of ts_symbollm (the CLI, the benchmark runner) depends on this
    interface rather than on any specific provider (e.g. Ollama) directly.

    To add a new backend: subclass this, implement `generate()` (and
    `pull()` if the runtime needs an explicit download step), and put it
    in its own module under `ts_symbollm/backends/` alongside `ollama.py`.
    No call site outside `ts_symbollm/backends/` should ever import a
    provider's SDK directly.
    '''

    @abstractmethod
    def generate(self, model: str, prompt: str, params: Optional[dict] = None) -> str:
        '''Generate a complete text response for the given prompt.'''
        raise NotImplementedError

    def pull(self, model: str) -> bool:
        '''
        Ensure a model is available locally, for backends that need an
        explicit pull/download step (e.g. a local runtime). Backends that
        don't need this (e.g. a hosted API) can rely on this default
        no-op implementation.
        '''
        return True

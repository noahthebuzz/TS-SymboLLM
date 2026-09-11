from __future__ import annotations

from typing import Iterator, Optional

import ollama
from ollama import GenerateResponse

from .base import Backend


class OllamaBackend(Backend):
    '''Backend implementation talking to a local Ollama server.'''

    def pull(self, model: str) -> bool:
        try:
            response = ollama.pull(model=model, stream=True)
            progress_states = set()
            print(f"[PULLING MODEL]: {model}")
            for progress in response:
                if progress.get('status') in progress_states:
                    continue
                progress_states.add(progress.get('status'))
                print(f"{progress.get('status')}")
            return True
        except Exception as e:
            print(f"Error: {e}")
            return False

    def generate(self, model: str, prompt: str, params: Optional[dict] = None) -> str:
        response = self._stream(model, prompt, params)
        return self._consume(response)

    @staticmethod
    def _stream(model: str, prompt: str, params: Optional[dict]) -> Iterator[GenerateResponse]:
        if params is not None:
            return ollama.generate(model=model, prompt=prompt, options=params, stream=True)
        return ollama.generate(model=model, prompt=prompt, stream=True)

    @staticmethod
    def _consume(response: Iterator[GenerateResponse]) -> str:
        response_string = ''
        try:
            for part in response:
                print(part['response'], end='', flush=True)
                response_string += part['response']
        except StopIteration:
            print('\n[ERROR]\n')
        print('\n\n[FINISHED]\n')
        return response_string

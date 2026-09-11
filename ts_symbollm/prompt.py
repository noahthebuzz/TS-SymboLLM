from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from .representation import Representation, apply as apply_representation


@dataclass
class _DataSection:
    context: str
    data: str


def _symbol_to_letter(value: float) -> str:
    return chr(ord('a') + int(round(value)))


class PromptBuilder:
    '''
    Assembles a prompt from a task description, one or more (context, data)
    sections, and a desired-output description -- replacing hand-authored
    prompt JSON files under prompts/single/... and prompts/multi/....

    A single data section renders the same way the old "single"-level
    prompts did (unlabeled "Data context:"/"Data:"); two or more sections
    render with numbered labels ("Data 1 context:", "Data 2 context:", ...),
    covering what the old "multi"-level prompts did.
    '''

    def __init__(self, task: str, desired_output: str):
        self.task = task
        self.desired_output = desired_output
        self._sections: List[_DataSection] = []

    def add_data(self, context: str, data: str) -> "PromptBuilder":
        '''Add a data section from already-formatted context/data text.'''
        self._sections.append(_DataSection(context=context, data=data))
        return self

    def add_series(
        self,
        context: str,
        points: Sequence[Tuple[str, float]],
        representation: Representation | str = Representation.RAW,
        **representation_kwargs,
    ) -> "PromptBuilder":
        '''
        Add a data section built directly from a time series
        ([(timestamp, value), ...], the shape used throughout the rest of
        ts_symbollm) and a chosen representation -- no intermediate JSON
        file needed.

        `representation_kwargs` are passed through to
        `representation.apply()` (e.g. `decimal_places`, `num_symbols`,
        `levels`).
        '''
        timestamps = [point[0] for point in points]
        values = [point[1] for point in points]
        represented = apply_representation(representation, values, timestamps, **representation_kwargs)
        data_text = self._format_series(represented, representation)
        return self.add_data(context=context, data=data_text)

    @staticmethod
    def _format_series(series: dict, representation: Representation | str) -> str:
        representation = Representation(representation)
        if representation is Representation.SYMBOLIC:
            return "".join(_symbol_to_letter(value) for value in series.values())
        return "\n".join(f"{timestamp}: {value}" for timestamp, value in series.items())

    def build(self) -> str:
        if not self._sections:
            raise ValueError("At least one data section is required (call add_data or add_series before build()).")

        prompt = f"Task:\n{self.task}\n"
        multi = len(self._sections) > 1
        for index, section in enumerate(self._sections, start=1):
            label = f" {index}" if multi else ""
            prompt += f"Data{label} context:\n{section.context}\nData{label}:\n{section.data}\n"
        prompt += f"Desired output format:\n{self.desired_output}\n"
        return prompt

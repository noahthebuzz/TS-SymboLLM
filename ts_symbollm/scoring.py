from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ScoreResult:
    '''A scorer's verdict on one response: a numeric score plus optional notes.'''

    score: float
    notes: Optional[str] = None


class Scorer(ABC):
    '''
    Minimal interface for scoring a model's response to a benchmark
    prompt, so the benchmark runner doesn't need to know whether scoring
    is heuristic, human-in-the-loop, or an LLM-judge -- only that it
    returns a ScoreResult.

    To add a new scorer (e.g. a human-in-the-loop or LLM-judge scorer
    later): subclass this, implement `score()`, and pass an instance
    wherever the benchmark runner accepts an optional scorer. Nothing
    about the runner or `ResultRecord` needs to change -- scoring is
    always optional, and a run works fine with no scorer at all.
    '''

    @abstractmethod
    def score(self, dataset_metadata: dict, prompt: str, response: str) -> ScoreResult:
        '''
        `dataset_metadata` carries whatever ground-truth context a given
        scorer needs (e.g. the expected trend for a synthetic dataset);
        its shape is scorer-specific by design, since different scorers
        need different context.
        '''
        raise NotImplementedError


@dataclass
class TrendPattern:
    '''
    Groups of keyword synonyms. A response "matches" a pattern when at
    least one keyword from *each* group appears in it (case-insensitive)
    -- e.g. a spike-then-decrease pattern needs both a spike/jump word
    and a decrease/decline word to count as mentioned.
    '''

    keyword_groups: List[List[str]]


KNOWN_TREND_PATTERNS = {
    "increasing": TrendPattern([
        ["increas", "rising", "rise", "climb", "upward", "higher", "warmer", "warming", "hotter", "escalat"],
    ]),
    "decreasing": TrendPattern([
        ["decreas", "declin", "drop", "fall", "lower", "cooler", "cooling", "colder", "reduc"],
    ]),
    "spike_then_decrease": TrendPattern([
        ["spike", "jump", "surge", "peak", "sudden"],
        ["decreas", "declin", "drop", "fall", "reduc", "return", "settl"],
    ]),
}


class TrendKeywordScorer(Scorer):
    '''
    Heuristic scorer for the synthetic showcase datasets (see
    examples/showcase/datagen.py): since their trend is synthetically
    controlled, we know the ground truth exactly, so we can check whether
    a response mentions the correct trend direction/phase transition
    without a human reading it.

    `dataset_metadata` must include a `"trend"` key naming one of
    `KNOWN_TREND_PATTERNS` -- e.g. `"increasing"` for
    `generate_temperature_tsd` (ramps from ~50C to ~95C), or
    `"spike_then_decrease"` for `generate_capacity_tsd` (jumps to ~75%
    then decreases back to ~30%).
    '''

    def score(self, dataset_metadata: dict, prompt: str, response: str) -> ScoreResult:
        trend = dataset_metadata.get("trend")
        pattern = KNOWN_TREND_PATTERNS.get(trend)
        if pattern is None:
            return ScoreResult(score=0.0, notes=f"Unknown or missing trend metadata: {trend!r}")

        response_lower = response.lower()
        missing_groups = [
            group for group in pattern.keyword_groups
            if not any(keyword in response_lower for keyword in group)
        ]

        if not missing_groups:
            return ScoreResult(score=1.0, notes=f"Mentioned all expected trend cues for {trend!r}.")
        return ScoreResult(score=0.0, notes=f"Missing expected cues for {trend!r}: {missing_groups}")

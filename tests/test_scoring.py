from ts_symbollm.scoring import KNOWN_TREND_PATTERNS, ScoreResult, Scorer, TrendKeywordScorer


class _AlwaysPassScorer(Scorer):
    def score(self, dataset_metadata, prompt, response):
        return ScoreResult(score=1.0, notes="always passes")


def test_scorer_subclass_satisfies_the_interface():
    scorer = _AlwaysPassScorer()
    result = scorer.score({}, "prompt", "response")
    assert isinstance(result, ScoreResult)
    assert result.score == 1.0


def test_trend_scorer_passes_when_increasing_cue_present():
    scorer = TrendKeywordScorer()
    result = scorer.score(
        {"trend": "increasing"},
        prompt="...",
        response="The temperature shows a clear rising trend over time.",
    )
    assert result.score == 1.0


def test_trend_scorer_fails_when_increasing_cue_absent():
    scorer = TrendKeywordScorer()
    result = scorer.score(
        {"trend": "increasing"},
        prompt="...",
        response="The temperature stays roughly constant throughout.",
    )
    assert result.score == 0.0
    assert "increasing" in result.notes


def test_trend_scorer_spike_then_decrease_requires_both_cues():
    scorer = TrendKeywordScorer()
    only_spike = scorer.score(
        {"trend": "spike_then_decrease"},
        prompt="...",
        response="Capacity suddenly spikes early on.",
    )
    assert only_spike.score == 0.0

    both_cues = scorer.score(
        {"trend": "spike_then_decrease"},
        prompt="...",
        response="Capacity spikes sharply, then gradually decreases back down.",
    )
    assert both_cues.score == 1.0


def test_trend_scorer_handles_unknown_trend_gracefully():
    scorer = TrendKeywordScorer()
    result = scorer.score({"trend": "not_a_real_trend"}, prompt="...", response="anything")
    assert result.score == 0.0
    assert "Unknown" in result.notes


def test_trend_scorer_handles_missing_metadata_gracefully():
    scorer = TrendKeywordScorer()
    result = scorer.score({}, prompt="...", response="anything")
    assert result.score == 0.0


def test_known_trend_patterns_cover_temperature_and_capacity_examples():
    # The two datasets explicitly named in the issue (#12).
    assert "increasing" in KNOWN_TREND_PATTERNS
    assert "spike_then_decrease" in KNOWN_TREND_PATTERNS

import pytest

from nightfocus.scoring import score_focus_quality


def test_perfect_ordering():
    # Perfect case: scores increase with distance from correct focus (2)
    scores = {0: 0.9, 1: 0.6, 2: 0.1, 3: 0.7, 4: 0.95}
    assert score_focus_quality(scores, correct_focus=2) == pytest.approx(0.0, abs=1e-6)


def test_equidistant_ordering():
    # Test with equidistant points - order within same distance shouldn't matter
    scores = {1: 0.2, 2: 0.1, 3: 0.15}  # 1 and 3 are equidistant from 2
    score1 = score_focus_quality(scores, correct_focus=2)

    # Same distances, different order within same distance group
    scores2 = {1: 0.15, 2: 0.1, 3: 0.2}
    score2 = score_focus_quality(scores2, correct_focus=2)

    assert score1 == pytest.approx(score2, abs=1e-6)


def test_worst_ordering():
    # Worst case: scores decrease with distance from correct focus
    scores = {0: 0.1, 1: 0.6, 2: 0.9, 3: 0.7, 4: 0.4}
    # Should be close to 1.0 (worst possible)
    assert score_focus_quality(scores, correct_focus=2) > 0.7


def test_empty_input():
    # Empty input should return 1.0 (worst score)
    assert score_focus_quality({}, correct_focus=0) == 1.0


def test_single_value():
    # Single value should be perfect (nothing to compare)
    assert score_focus_quality({0: 0.5}, correct_focus=0) == 0.0


def test_duplicate_distances():
    # Test with multiple points at the same distance
    scores = {0: 0.3, 1: 0.2, 2: 0.1, 3: 0.15, 4: 0.25}
    # 0 and 4 are equidistant from 2, as are 1 and 3
    score = score_focus_quality(scores, correct_focus=2)
    assert 0.0 <= score <= 1.0


def test_almost_perfect_ordering():
    # Almost perfect ordering - one pair is out of order
    scores = {0: 0.9, 1: 0.5, 2: 0.1, 3: 0.7, 4: 0.8}
    # Should be better than random but not perfect
    score = score_focus_quality(scores, correct_focus=2)
    assert 0.0 <= score < 0.5


def test_ordering_quality_decreases_with_disorder():
    # Start with perfect ordering
    perfect = {0: 0.9, 1: 0.6, 2: 0.1, 3: 0.7, 4: 0.95}
    perfect_score = score_focus_quality(perfect, correct_focus=2)
    assert perfect_score == pytest.approx(0.0, abs=1e-6)

    # Slight disorder
    slight_disorder = {0: 0.9, 1: 0.6, 2: 0.7, 3: 0.1, 4: 0.95}
    slight_score = score_focus_quality(slight_disorder, correct_focus=2)
    assert slight_score > perfect_score

    # More disorder - move the worst focus to be best
    more_disorder = {0: 0.9, 1: 0.6, 2: 0.95, 3: 0.7, 4: 0.1}
    more_score = score_focus_quality(more_disorder, correct_focus=2)
    assert more_score > slight_score

    # Verify all scores are in valid range
    assert 0.0 <= slight_score <= 1.0
    assert 0.0 <= more_score <= 1.0

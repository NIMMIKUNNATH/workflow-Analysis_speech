import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from row_assignment import assign_words_to_intervals, best_interval_index


def word(start, end, text):
    return {"start": start, "end": end, "text": text}


def intervals():
    return [
        {"start": 0.0, "end": 1.0, "speaker": "A"},
        {"start": 1.0, "end": 2.0, "speaker": "B"},
    ]


def test_every_eligible_word_receives_exactly_one_assignment():
    words = [word(0.2, 0.5, "one"), word(0.8, 1.4, "two")]
    rows, unassigned = assign_words_to_intervals(words, intervals())
    assert not unassigned
    assert sum(len(row) for row in rows) == len(words)


def test_boundary_tie_is_deterministic():
    crossing = word(0.5, 1.5, "tie")
    choices = [best_interval_index(crossing, intervals()) for _ in range(20)]
    assert choices == [0] * 20


def test_no_word_can_appear_twice_in_constructed_rows():
    crossing = word(0.8, 1.4, "once")
    rows, _ = assign_words_to_intervals([crossing], intervals())
    emitted = [item for row in rows for item in row]
    assert emitted == [crossing]


def test_assigned_total_equals_eligible_hypothesis_words():
    words = [
        word(0.2, 0.4, "one"),
        word(0.8, 1.4, "two"),
        word(2.5, 2.8, "outside"),
    ]
    rows, unassigned = assign_words_to_intervals(words, intervals())
    assert len(unassigned) == 1
    assert sum(len(row) for row in rows) == len(words) - len(unassigned)


def test_overlapping_intervals_still_emit_each_word_once():
    overlapping = [
        {"start": 0.0, "end": 1.5, "speaker": "A"},
        {"start": 1.0, "end": 2.0, "speaker": "B"},
    ]
    crossing = word(1.1, 1.4, "overlap")
    rows, unassigned = assign_words_to_intervals([crossing], overlapping)
    assert not unassigned
    assert [item for row in rows for item in row] == [crossing]


def test_empty_hypothesis_is_valid():
    rows, unassigned = assign_words_to_intervals([], intervals())
    assert rows == [[], []]
    assert unassigned == []


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
        print(f"PASS  {test.__name__}")
    print(f"\nPASSED: {len(tests)}")

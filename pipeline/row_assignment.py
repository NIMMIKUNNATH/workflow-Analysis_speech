"""Deterministic assignment of timestamped words to retained intervals."""


def overlap_duration(start1, end1, start2, end2):
    return max(0.0, min(end1, end2) - max(start1, start2))


def best_interval_index(word, intervals):
    """Return the uniquely selected interval index, or ``None`` if ineligible.

    Selection uses greatest temporal overlap, then the earliest interval in
    the retained interval sequence. Python's ``max`` is stable, so an exact
    overlap tie is resolved in favour of the first candidate.
    """
    candidates = []
    for index, interval in enumerate(intervals):
        overlap = overlap_duration(
            word["start"], word["end"], interval["start"], interval["end"]
        )
        if overlap <= 0:
            continue
        candidates.append((overlap, -index, index))
    return max(candidates)[-1] if candidates else None


def assign_words_to_intervals(words, intervals):
    """Partition eligible words across intervals without duplication or loss."""
    assigned = [[] for _ in intervals]
    unassigned = []
    for word in words:
        index = best_interval_index(word, intervals)
        if index is None:
            unassigned.append(word)
        else:
            assigned[index].append(word)

    assigned_count = sum(len(row) for row in assigned)
    eligible_count = len(words) - len(unassigned)
    if assigned_count != eligible_count:
        raise AssertionError(
            f"word-conservation failure: {eligible_count} eligible, "
            f"{assigned_count} assigned"
        )
    if len({id(word) for row in assigned for word in row}) != assigned_count:
        raise AssertionError("a hypothesis word was assigned more than once")
    return assigned, unassigned

"""Repeating the training rows of small lines of business (training.corpus_view.line_repeats)."""

from training.corpus_view import _balance, line_repeats


def _rows(counts):
    return [{"source_id": f"{lob}-{i}", "lob": [lob]} for lob, n in counts.items() for i in range(n)]


DELIVERED = {"homeowners": 593, "recreational_vehicle": 208, "dwelling_fire": 176, "personal_auto": 143,
             "classic_auto": 77, "personal_umbrella": 55, "ocean_marine": 44, "motorcycle": 33}


def test_small_lines_are_repeated_toward_the_floor_large_ones_not():
    repeats = line_repeats(_rows(DELIVERED), min_documents=100, max_repeat=3)
    assert repeats["homeowners"] == 1 and repeats["personal_auto"] == 1
    assert repeats["classic_auto"] == 2 and repeats["personal_umbrella"] == 2
    assert repeats["ocean_marine"] == 3 and repeats["motorcycle"] == 3      # capped at 3


def test_when_every_line_is_small_nothing_is_repeated():
    """A smoke corpus: tripling everything would triple the run and balance nothing."""
    repeats = line_repeats(_rows({"homeowners": 11, "motorcycle": 11, "personal_auto": 11}),
                           min_documents=100, max_repeat=3)
    assert set(repeats.values()) == {1}


def test_documents_not_rows_decide_how_small_a_line_is():
    rows = [{"source_id": "h1", "lob": ["homeowners"]}] * 500 + _rows({"motorcycle": 150})
    assert line_repeats(rows, min_documents=100, max_repeat=3)["homeowners"] == 3


def test_off_when_configured_off():
    assert set(line_repeats(_rows(DELIVERED), min_documents=0, max_repeat=3).values()) == {1}


def test_rows_are_written_as_many_times_as_their_line():
    import json

    body = "\n".join(json.dumps(r) for r in _rows({"homeowners": 2, "motorcycle": 1}))
    out, count = _balance(body, {"homeowners": 1, "motorcycle": 3})
    assert count == 5 and out.count('"motorcycle"') == 3

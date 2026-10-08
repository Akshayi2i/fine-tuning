"""Fixtures shared across test modules."""

from __future__ import annotations

import pytest


@pytest.fixture
def no_window_overlap(monkeypatch):
    """Plan windows with no page shared between the windows of a split run.

    For tests of what happens at a window boundary - a fragment left without
    its identifier, a link to a row the window does not hold, two halves of a
    row joined by the merge. The configured overlap (``run_overlap_pages``)
    moves those boundaries; the mechanisms they test do not go away with it.
    """
    import common.schema_sections

    monkeypatch.setattr(common.schema_sections, "run_overlap", lambda lob=None: 0)

"""Whether a value is printed, and on which pages: one answer for every caller.

The matching the data audit has always used (``data_pipeline.audit``): case,
punctuation and line breaks are ignored, and a value is on a page when it
appears there as a phrase on word boundaries or, across line breaks and table
columns, as every one of its words. A value shorter than
:data:`MIN_CHECKABLE_CHARS` ("1", "NY") is on almost every page and is not
checked at all - it is neither grounded nor ungrounded.

One definition, because four places ask the same question and each had its own
answer: the audit and label verification matched words, the hallucination
metric and the calibration feature matched a bare substring of the whole text -
so "1" was always "found", and a wrong amount was never compared in the form
the page prints. :func:`ground` adds the page: printed on a page the value
cites, printed only on others, printed on no page sent, or not checkable.

Pure: no I/O, no heavy imports, safe for serving.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

#: Values this short ("Y", "1") match almost any page; they are not checked.
MIN_CHECKABLE_CHARS = 3

_NON_WORD = re.compile(r"[^0-9a-z]+")

#: :class:`Grounding` statuses.
ON_CITED_PAGE = "on_cited_page"
ON_OTHER_PAGE = "on_other_page"
NOT_PRINTED = "not_printed"
UNCHECKED = "unchecked"


def normalise(text: Any) -> str:
    """Case, punctuation and line breaks removed; words separated by one space."""
    return " ".join(_NON_WORD.sub(" ", str(text).casefold()).split())


_FIGURE = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")
#: Only figure characters: an amount or a count, not a code that holds digits.
_AMOUNT = re.compile(r"[\s$()+\-.,%]*\d[\d,]*(?:\.\d+)?[\s$()%]*")


class PageText:
    """A page's text normalised once, for every value looked up on it."""

    def __init__(self, text: str | None) -> None:
        self.padded = f" {normalise(text or '')} "
        self.words = set(self.padded.split())
        #: Every amount the page prints, as a number: "$7,579.67" and "7579.67"
        #: are one. Not a bare digit or two ("2 TOWN RD", "Page 3"): "$2.00"
        #: would be printed on every page - but "$25" is an amount.
        self.numbers = {number for number in map(_as_number, (
            figure for figure in _FIGURE.findall(text or "")
            if len(figure) >= 3 or any(mark in figure for mark in "$.,"))) if number is not None}


def _as_number(text: Any) -> float | None:
    """An amount or a count as a number (2 decimals), when ``text`` is only that."""
    if isinstance(text, bool):
        return None
    if isinstance(text, (int, float)):
        return round(float(text), 2)
    if not isinstance(text, str) or not _AMOUNT.fullmatch(text):
        return None
    figures = re.sub(r"[^\d.]", "", text)
    try:
        return round(float(figures), 2)
    except ValueError:
        return None


def appears_on(value: str, page: PageText) -> bool:
    """Whether an already-normalised ``value`` is on ``page``: as a phrase, or -
    a phrase broken across lines or table columns - as every one of its words.
    Not so for a code of short pieces: "HX-0000-0001" is the words "hx", "0000"
    and "0001", which any page holding "HX-0000-0000" and a "0001" also holds.
    Only a value with a word of three letters or more is matched word by word."""
    if not value:
        return False
    if f" {value} " in page.padded:
        return True
    words = value.split()
    return (len(words) > 1 and any(sum(ch.isalpha() for ch in word) >= 3 for word in words)
            and set(words) <= page.words)


def on_page(raw: Any, page: PageText) -> bool:
    """Whether ``raw`` is printed on ``page``: as a phrase or every one of its
    words, or - an amount or a count - as the same figure in another format."""
    if appears_on(normalise(raw), page):
        return True
    number = _as_number(raw)
    return number is not None and number in page.numbers


def appears(raw: Any, page_text: str | None) -> bool:
    """Whether ``raw`` is on the page, as a phrase or - across line breaks and
    columns - as every one of its words, or as the same figure."""
    return on_page(raw, PageText(page_text))


def checkable(raw: Any) -> bool:
    """Whether ``raw`` is long enough to be looked for at all."""
    return raw is not None and len(normalise(raw)) >= MIN_CHECKABLE_CHARS


@dataclass(frozen=True)
class Grounding:
    """Where a value is printed, against the pages it cites."""

    status: str
    #: The pages that print it, among those searched.
    pages: tuple[int, ...] = ()

    @property
    def printed(self) -> bool:
        return self.status in (ON_CITED_PAGE, ON_OTHER_PAGE)


def ground(raw: Any, cited: Iterable[Any] | None, pages: Mapping[int, str | PageText | None]) -> Grounding:
    """Where ``raw`` is printed among ``pages`` (page number -> its text).

    * :data:`ON_CITED_PAGE` - on a page it cites (or, citing none, on any page);
    * :data:`ON_OTHER_PAGE` - only on pages it does not cite;
    * :data:`NOT_PRINTED` - on no page searched;
    * :data:`UNCHECKED` - too short to look for, no page has text, or a page it
      cites has no text (a scan the OCR could not read may still print it).
    """
    value = normalise(raw) if raw is not None else ""
    if len(value) < MIN_CHECKABLE_CHARS:
        return Grounding(UNCHECKED)
    texts = {int(number): text if isinstance(text, PageText) else PageText(text)
             for number, text in pages.items()}
    if not any(text.words for text in texts.values()):
        return Grounding(UNCHECKED)
    found = tuple(sorted(number for number, text in texts.items() if on_page(raw, text)))
    wanted = {int(page) for page in cited or () if str(page).lstrip("-").isdigit()}
    if found:
        return Grounding(ON_CITED_PAGE if not wanted or wanted & set(found) else ON_OTHER_PAGE, found)
    if any(number in texts and not texts[number].words for number in wanted):
        return Grounding(UNCHECKED)
    return Grounding(NOT_PRINTED)

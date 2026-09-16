"""Near-duplicate detection and ``group_id`` assignment (arch v2.1 §8.2).

Insurance documents come in **families**. The same carrier's template, the same
account renewed every year, the same agency issuing hundreds of certificates off
one form. Splitting by source document puts one member of a family in ``train``
and another in ``test``, and the eval number that comes back measures how well
the model memorised a template — not whether it generalises to an unseen one.

That is the failure this module exists to prevent, and it is invisible: nothing
crashes, the metrics simply come back better than the model deserves. The v1
pipeline split at ``source_id`` level, which stopped modality variants leaking
but never addressed families at all.

Three signals, each catching a different kind of sameness:

* **SHA-256** over the raw bytes — the same file ingested twice.
* **Perceptual hash of the page-1 layout** — the same template, different data.
  A renewal and its prior year differ in every value and not at all in shape.
* **MinHash over the OCR text** — near-duplicate content that the layout hash
  misses, such as a re-issued certificate with one address corrected.

Near-duplicates are **assigned the same group, not dropped**. A carrier that
issues four hundred near-identical certificates is a real part of the
distribution; removing them would train the model on a corpus that does not look
like production. What must not happen is those four hundred spanning the split.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

#: Shingle width for MinHash. Five words is long enough that ordinary insurance
#: boilerplate ("the named insured shown in the declarations") does not collide
#: across unrelated documents, and short enough to survive an OCR error inside
#: an otherwise identical paragraph.
SHINGLE_WORDS = 5

#: MinHash permutations. 128 gives a Jaccard estimate within a few percent,
#: which is ample when the decision is a threshold rather than a ranking.
MINHASH_PERMUTATIONS = 128

#: Estimated Jaccard at or above which two documents are near-duplicates.
#: Deliberately high: merging two groups that are not really the same family
#: costs test-set diversity, which is harder to notice than the reverse.
NEAR_DUPLICATE_THRESHOLD = 0.85

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9 ]")


class GroupingError(RuntimeError):
    """Raised when documents cannot be grouped safely."""


@dataclass
class DocumentFingerprint:
    """Everything about one document that decides which family it belongs to."""

    source_id: str
    doc_type: str
    content_sha256: str
    layout_phash: str | None = None
    minhash: tuple[int, ...] = ()

    #: The three fields that name a family. Any of them may be missing — a
    #: scanned certificate often names no account — and the grouping degrades
    #: gracefully rather than refusing.
    carrier: str | None = None
    template_id: str | None = None
    account: str | None = None

    def identity_key(self) -> str | None:
        """The declared family, before any hash evidence is considered.

        ``None`` when nothing is declared. An absent identity is not a shared
        identity: keying every unattributed document to the same placeholder
        merged all of them into one enormous family, which would have put the
        whole corpus on one side of the split.
        """
        parts = [_canon(self.carrier), _canon(self.template_id), _canon(self.account)]
        if not any(parts):
            return None
        return "|".join(p or "-" for p in parts)


@dataclass
class GroupingReport:
    """Which documents grouped with which, and on what evidence."""

    group_of: dict[str, str] = field(default_factory=dict)
    members: dict[str, list[str]] = field(default_factory=dict)

    #: source_id -> the source_id it exactly duplicates. Reported, never dropped:
    #: an operator needs to know an upload happened twice.
    exact_duplicates: dict[str, str] = field(default_factory=dict)

    #: (a, b, jaccard) for pairs merged on text or layout similarity.
    near_duplicate_pairs: list[tuple[str, str, float]] = field(default_factory=list)

    #: Documents naming no carrier at all. They can still be grouped by layout
    #: and text, but the held-out-carrier slice cannot use them.
    unattributed: list[str] = field(default_factory=list)

    @property
    def group_count(self) -> int:
        return len(self.members)

    def largest_groups(self, limit: int = 5) -> list[tuple[str, int]]:
        return sorted(
            ((g, len(m)) for g, m in self.members.items()), key=lambda kv: -kv[1]
        )[:limit]

    def as_dict(self) -> dict[str, Any]:
        return {
            "group_count": self.group_count,
            "documents": len(self.group_of),
            "exact_duplicates": dict(sorted(self.exact_duplicates.items())),
            "near_duplicate_pairs": [
                {"a": a, "b": b, "jaccard": round(j, 3)}
                for a, b, j in sorted(self.near_duplicate_pairs)
            ],
            "unattributed": sorted(self.unattributed),
            "largest_groups": [{"group_id": g, "documents": n} for g, n in self.largest_groups()],
        }


def _canon(value: str | None) -> str | None:
    """Casefold and strip punctuation, so ``Acme Ins. Co.`` and ``ACME INS CO``
    are one carrier rather than two families."""
    if not value:
        return None
    text = _NON_ALNUM.sub(" ", _WS.sub(" ", str(value)).strip().casefold())
    return _WS.sub(" ", text).strip() or None


def shingles(text: str, width: int = SHINGLE_WORDS) -> set[str]:
    """Overlapping word n-grams, normalised.

    Word-level rather than character-level: an OCR error corrupts a character in
    a way that breaks every character shingle overlapping it, while a word-level
    shingle loses only the shingles containing that word.
    """
    words = _NON_ALNUM.sub(" ", _WS.sub(" ", text or "").casefold()).split()
    if len(words) < width:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + width]) for i in range(len(words) - width + 1)}


def minhash(text: str, permutations: int = MINHASH_PERMUTATIONS) -> tuple[int, ...]:
    """A MinHash signature over the document's OCR text.

    Each permutation is a differently-salted SHA-256, which is slower than the
    usual multiply-shift family and needs no tuning to behave — the corpus is
    thousands of documents, not billions, so correctness beats throughput here.
    """
    grams = shingles(text)
    if not grams:
        return ()
    signature = []
    for i in range(permutations):
        salt = str(i).encode()
        signature.append(min(
            int.from_bytes(hashlib.sha256(salt + g.encode()).digest()[:8], "big")
            for g in grams
        ))
    return tuple(signature)


def estimated_jaccard(a: Sequence[int], b: Sequence[int]) -> float:
    """Share of matching positions — the MinHash estimate of set similarity."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x == y for x, y in zip(a, b, strict=True)) / len(a)


def assign_groups(
    fingerprints: Iterable[DocumentFingerprint],
    *,
    near_duplicate_threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> GroupingReport:
    """Assign every document a ``group_id``. All members of a family share one.

    Grouping is transitive by construction: A near-duplicates B and B
    near-duplicates C puts all three in one group even where A and C fall under
    the threshold. That is deliberate — a chain of renewals is one account, and
    letting the ends of the chain split would leak exactly what this prevents.
    """
    documents = sorted(fingerprints, key=lambda f: f.source_id)
    report = GroupingReport()
    if not documents:
        return report

    # Union-find over documents. Start from the declared identity, then merge on
    # hash evidence, so a stated carrier+template+account always groups even when
    # the text similarity is low (two very different claims on one account).
    parent: dict[str, str] = {d.source_id: d.source_id for d in documents}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_identity: dict[str, list[str]] = {}
    by_content: dict[str, str] = {}
    by_layout: dict[str, list[str]] = {}

    for doc in documents:
        identity = doc.identity_key()
        if identity is not None:
            by_identity.setdefault(identity, []).append(doc.source_id)
        if doc.carrier is None:
            report.unattributed.append(doc.source_id)

        # Exact duplicate: same bytes. Recorded and grouped, never dropped — the
        # operator needs to know the same file arrived twice.
        if doc.content_sha256 in by_content:
            first = by_content[doc.content_sha256]
            report.exact_duplicates[doc.source_id] = first
            union(doc.source_id, first)
        else:
            by_content[doc.content_sha256] = doc.source_id

        if doc.layout_phash:
            by_layout.setdefault(doc.layout_phash, []).append(doc.source_id)

    # A declared identity is the strongest signal — it is what a human recorded.
    for members in by_identity.values():
        for other in members[1:]:
            union(members[0], other)

    # Identical page-1 layout is the same template, whatever the values say.
    for members in by_layout.values():
        for other in members[1:]:
            union(members[0], other)

    # Text similarity, pairwise. Quadratic, which is fine at corpus scale and
    # honest about it — a banded LSH index is the change to make if this ever
    # runs on hundreds of thousands of documents.
    with_text = [d for d in documents if d.minhash]
    for i, a in enumerate(with_text):
        for b in with_text[i + 1:]:
            if find(a.source_id) == find(b.source_id):
                continue
            similarity = estimated_jaccard(a.minhash, b.minhash)
            if similarity >= near_duplicate_threshold:
                report.near_duplicate_pairs.append((a.source_id, b.source_id, similarity))
                union(a.source_id, b.source_id)

    for doc in documents:
        root = find(doc.source_id)
        group_id = f"grp-{hashlib.sha256(root.encode()).hexdigest()[:12]}"
        report.group_of[doc.source_id] = group_id
        report.members.setdefault(group_id, []).append(doc.source_id)

    for members in report.members.values():
        members.sort()

    if report.exact_duplicates:
        log.warning(
            "%d exact duplicate(s) ingested; grouped rather than dropped so the corpus still "
            "reflects production, but check whether an upload ran twice: %s",
            len(report.exact_duplicates), sorted(report.exact_duplicates)[:5],
        )
    log.info(
        "grouped %d documents into %d families; largest %s",
        len(report.group_of), report.group_count, report.largest_groups(3),
    )
    return report

"""An answer's tokens at corpus build, as the pre-launch length check counts them
(data_pipeline.dataset_builder.cap_check.target_tokens)."""

from __future__ import annotations

import math

from data_pipeline.dataset_builder import cap_check


class _Tokenizer:
    """A stand-in for the base model's tokenizer: ``per_char`` tokens a character."""

    def __init__(self, per_char: float) -> None:
        self.per_char = per_char

    def __call__(self, text, add_special_tokens=True):
        assert add_special_tokens is False                     # the answer alone, as training counts it
        return {"input_ids": list(range(int(len(text) * self.per_char)))}


def test_an_answer_is_counted_by_the_models_tokenizer_when_it_is_here(monkeypatch):
    monkeypatch.setattr(cap_check, "_answer_tokenizer", lambda: _Tokenizer(0.5))
    assert cap_check.target_tokens("x" * 100) == 50


def test_without_the_tokenizer_an_answer_estimate_is_padded(monkeypatch):
    monkeypatch.setattr(cap_check, "_answer_tokenizer", lambda: None)
    text = '{"raw": "$1,000", "parsed": 1000, "page_ref": [3]}'
    assert cap_check.target_tokens(text) == math.ceil(cap_check.estimate_text_tokens(text) * 1.25)


def test_an_answer_over_its_reservation_by_the_real_count_is_refused_where_the_estimate_fit(monkeypatch):
    """The smoke run: answers estimated under 12,288 were 12,954-13,492 tokens,
    and training refused them at launch."""
    target = '{"rows": "' + "a" * 35_000 + '"}'
    assert cap_check.estimate_text_tokens(target) < 12_288
    monkeypatch.setattr(cap_check, "_answer_tokenizer", lambda: _Tokenizer(0.4))   # 14,000 real
    estimate = cap_check.estimate_row(task="policy_schedule", system_prompt="schema", ocr_pages=None,
                                      page_count=1, target_json=target, doc_type="policy")
    fits, _cap, why = cap_check.evaluate(estimate)
    assert not fits and "12,288" in why

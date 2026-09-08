"""The pilot validation protocol (SPEC_15, arch §16).

Three experiments in strictly increasing order of investment, each answering a
different question and each cheap relative to the next:

* **A — zero-shot baseline** (no annotation cost): how much does the untuned base
  model already extract, and *where* does it fail?
* **B — smoke test** (5 documents per type): does the code pipeline work end to
  end? A failure here is a bug, not an architecture problem.
* **C — pilot run** (25–30 per type): does the architecture's generalisation
  claim hold on held-out documents?

**Run them in order.** Skipping to C spends the annotation budget before knowing
whether the prompt, the schema, or the base model is the thing that needs fixing.

Nothing in the repo imports this package — it is a runbook that calls the
libraries, never a library other code depends on. In particular every experiment
routes through ``serving/pipeline.py`` by way of ``testing/run_extraction.py``,
so pilot numbers are comparable to production numbers rather than measurements of
a second implementation.
"""

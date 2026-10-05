"""The human review interface (IMPL-04 §4).

Generates the labeling configuration for an external tool (Label Studio's XML
config is the reference target; Argilla consumes the same field list), and
adapts tasks in and annotations out.

**Why this is a deliverable and not a preference.** Reviewing an insurance
document means deciding which canonical field a value belongs to, and that
decision is exactly the one the model has to learn. If three reviewers disagree
about whether a document's *Applicant* is `insured_name`, the disagreement is not
noise around a correct answer — it trains directly into the model as the answer.
So the review surface carries the same semantic glosses the prompt does, taken
from the same schema `description` fields, and it records the surface label the
reviewer actually saw.

**The alias registry is shown here, and only here.** Surfacing it to a *human*
is the opposite of the master §1.4 anti-pattern: the rule forbids a runtime
lookup table doing the model's semantic mapping, because that caps the system at
a hand-written list. A reviewer reading the same list to label consistently is
what makes the corpus teach the mapping in the first place. `test_no_runtime_aliases`
scans `serving/`, `inference_core/` and `testing/` — not this package — for
precisely that reason.
"""

from data_pipeline.labeling.review_tool.config import (
    ReviewField,
    build_labeling_config,
    review_fields_for,
)
from data_pipeline.labeling.review_tool.tasks import (
    ReviewTask,
    completed_to_golden,
    double_annotation_sample,
    task_from_document,
)

__all__ = [
    "ReviewField",
    "ReviewTask",
    "build_labeling_config",
    "completed_to_golden",
    "double_annotation_sample",
    "review_fields_for",
    "task_from_document",
]

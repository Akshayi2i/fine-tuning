# SPEC_10 — Domain Sentence Encoder Fine-Tuning

**Owner:** ML Engineering  
**Depends on:** SPEC_04 (DomainSentenceEncoder consumer)  
**Language:** Python 3.11 · sentence-transformers · datasets  
**Run on:** CPU or single GPU (lightweight) — offline, Phase 3  
**Output:** Fine-tuned encoder saved to `models/domain_sentence_encoder/`  
**NOT the DAPT BPE tokeniser** — see SPEC_14 for DAPT  

---

## 1. Purpose

The Domain Sentence Encoder is a lightweight embedding model used by the Section Boundary
Chunker (SPEC_04) to confirm candidate chunk boundaries via cosine similarity. A boundary
is confirmed when cosine similarity between adjacent text segments is below 0.35 (low
similarity = genuinely different topics).

The base model (`all-MiniLM-L6-v2`, 22M params) already understands sentence semantics,
but lacks P&C insurance vocabulary. Fine-tuning on insurance corpora teaches it that
"Claimant: John Smith — Date of Loss: 01/15/2023" and "POLICY YEAR: 2022–2023" are
semantically different section types (boundary), while consecutive claim rows in the same
policy year are semantically similar (no boundary).

---

## 2. Training approach

**Loss:** `MultipleNegativesRankingLoss` (contrastive, efficient, no explicit negatives needed)  
**Training pairs:** (anchor, positive) where anchor and positive are text segments from the
SAME section of a document. Negatives are other anchors in the same batch (in-batch negatives).

During inference, a low cosine similarity between adjacent segments indicates a boundary —
the model learns to embed same-section text closer together in embedding space.

---

## 3. Training data construction (`scripts/build_encoder_dataset.py`)

```python
"""
Construct sentence encoder training pairs from de-identified loss run archive.

Input: de-identified markdown files (from MinerU) tagged with section boundaries
Output: data/encoder/pairs.jsonl

Pair types:
  POSITIVE: two consecutive paragraphs from the SAME policy-year section
  [In-batch negatives]: paragraphs from DIFFERENT sections in the same document

Pairs are constructed ONLY from documents where section boundaries are
unambiguous (confirmed by a human annotator or heuristic with high confidence).
"""

import json, random
from pathlib import Path

def build_pairs(markdown_dir: str, output_path: str, target_pairs: int = 5000):
    pairs = []
    md_files = list(Path(markdown_dir).glob("*.md"))

    for md_file in md_files:
        sections = parse_sections(md_file)
        # sections: list of {"heading": str, "paragraphs": list[str], "year": str}

        for section in sections:
            paras = section["paragraphs"]
            if len(paras) < 2:
                continue
            # Create positive pairs from consecutive paragraphs in the same section
            for i in range(len(paras) - 1):
                anchor = paras[i]
                positive = paras[i + 1]
                if len(anchor.strip()) > 30 and len(positive.strip()) > 30:
                    pairs.append({
                        "anchor": anchor,
                        "positive": positive,
                        "section": section["heading"],
                    })
            if len(pairs) >= target_pairs:
                break
        if len(pairs) >= target_pairs:
            break

    random.shuffle(pairs)
    with open(output_path, "w") as f:
        for pair in pairs:
            f.write(json.dumps(pair) + "\n")

    print(f"Built {len(pairs)} training pairs from {len(md_files)} documents")


def parse_sections(md_file: Path) -> list[dict]:
    """
    Split a markdown file into sections at heading boundaries (## or ###).
    Each section: {heading, paragraphs, year}.
    """
    content = md_file.read_text()
    sections = []
    current_heading = "PREAMBLE"
    current_paras = []

    for line in content.splitlines():
        if line.startswith("## ") or line.startswith("### "):
            if current_paras:
                sections.append({"heading": current_heading, "paragraphs": current_paras})
            current_heading = line.strip("# ").strip()
            current_paras = []
        elif line.strip():
            current_paras.append(line.strip())

    if current_paras:
        sections.append({"heading": current_heading, "paragraphs": current_paras})
    return sections
```

---

## 4. Training script (`scripts/train_encoder.py`)

```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from torch.utils.data import DataLoader
import json

BASE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_PATH = "models/domain_sentence_encoder"
DATA_PATH   = "data/encoder/pairs.jsonl"

TRAIN_CONFIG = {
    "num_epochs":     5,
    "batch_size":     32,
    "warmup_steps":   100,
    "learning_rate":  2e-5,
    "weight_decay":   0.01,
    "show_progress_bar": True,
}

def main():
    model = SentenceTransformer(BASE_MODEL)

    # Load pairs
    examples = []
    with open(DATA_PATH) as f:
        for line in f:
            d = json.loads(line)
            examples.append(InputExample(texts=[d["anchor"], d["positive"]]))

    # 80/20 split
    split = int(len(examples) * 0.8)
    train_examples = examples[:split]
    val_examples   = examples[split:]

    # DataLoader
    train_dataloader = DataLoader(
        train_examples,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
    )

    # Loss: MultipleNegativesRankingLoss
    # No explicit negatives needed — other anchors in the batch serve as negatives
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Evaluator: cosine similarity on val pairs (same-section → high similarity)
    evaluator = build_evaluator(val_examples)

    # Train
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        evaluator=evaluator,
        evaluation_steps=200,
        epochs=TRAIN_CONFIG["num_epochs"],
        warmup_steps=TRAIN_CONFIG["warmup_steps"],
        optimizer_params={"lr": TRAIN_CONFIG["learning_rate"]},
        output_path=OUTPUT_PATH,
        save_best_model=True,
        show_progress_bar=True,
    )

    print(f"Encoder saved to {OUTPUT_PATH}")
    evaluate_boundary_detection(model)


def build_evaluator(val_examples: list[InputExample]):
    """
    Build an EmbeddingSimilarityEvaluator that measures cosine similarity
    for same-section pairs (should be HIGH) vs cross-section pairs (should be LOW).
    We label same-section pairs as similarity=1.0 and generate cross-section
    pairs with similarity=0.0 from shuffled examples.
    """
    sentences1 = [e.texts[0] for e in val_examples]
    sentences2 = [e.texts[1] for e in val_examples]
    scores     = [1.0] * len(val_examples)

    # Add cross-section pairs (anchor[i], positive[j] where i≠j) with label=0.0
    import random
    for i in range(len(val_examples)):
        j = random.choice([k for k in range(len(val_examples)) if k != i])
        sentences1.append(val_examples[i].texts[0])
        sentences2.append(val_examples[j].texts[1])
        scores.append(0.0)

    return EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)


def evaluate_boundary_detection(model: SentenceTransformer):
    """
    Evaluate boundary detection on held-out loss run fixtures.
    For each known boundary in the fixture, check that cosine similarity < 0.35.
    For each known non-boundary, check that cosine similarity > 0.60.
    """
    BOUNDARY_THRESHOLD = 0.35
    NON_BOUNDARY_THRESHOLD = 0.60

    fixtures = [
        # (text_before_boundary, text_after_boundary, is_boundary)
        ("POLICY YEAR: 2021-2022\n\nTotal claims: 12\nTotal incurred: $45,230.00",
         "POLICY YEAR: 2022-2023\n\nClaim #: 2023-001\nClaimant: ...",
         True),
        ("Claim #: 2022-005\nClaimant: Jones\nDate of Loss: 03/15/2022",
         "Claim #: 2022-006\nClaimant: Smith\nDate of Loss: 04/22/2022",
         False),
    ]

    correct = 0
    for before, after, expected_boundary in fixtures:
        emb_before = model.encode(before, normalize_embeddings=True)
        emb_after  = model.encode(after,  normalize_embeddings=True)
        cosine = float(emb_before @ emb_after)
        predicted_boundary = cosine < BOUNDARY_THRESHOLD
        if predicted_boundary == expected_boundary:
            correct += 1
        print(f"cosine={cosine:.3f} expected_boundary={expected_boundary} predicted={predicted_boundary}")

    print(f"Boundary detection accuracy: {correct}/{len(fixtures)}")
```

---

## 5. Evaluation metrics for promotion

```
Target metrics (measured on 50 held-out boundary/non-boundary pairs):
  Boundary detection precision ≥ 0.85
  Boundary detection recall    ≥ 0.80
  F1                           ≥ 0.82

Cosine similarity statistics on val set:
  Same-section pairs:  mean cosine ≥ 0.70
  Cross-section pairs: mean cosine ≤ 0.40
```

---

## 6. Dataset source and de-identification

```
Source: De-identified loss run markdown files in Azure Blob (Zone B)
        Generated by MinerU (SPEC_05) from Presidio-processed PDFs
Volume: ~200 documents × ~25 sections each = ~5,000 sections → ~25,000 pairs
        (4,000 pairs per document type; 1,000 reserved for held-out eval)

De-identification applied BEFORE any encoder training:
  - Named claimants → "CLAIMANT_001", "CLAIMANT_002" etc.
  - Employer names → "EMPLOYER_001"
  - Policy numbers → "POL-XXXX-2022"
  - Dollar amounts → KEPT (amounts are structurally important for boundary detection)
  - Dates → KEPT (date ranges identify policy years)
  - Carrier names → KEPT (in headers, help identify section type)

Student researchers may access encoder training data after Presidio de-identification.
Raw PII never leaves Zone A.
```

---

## 7. Serving integration

The fine-tuned encoder is loaded once at Pipeline API startup:

```python
# In fideon/chunker/sentence_encoder.py (SPEC_04):
FINE_TUNED_MODEL_PATH = "models/domain_sentence_encoder"

class DomainSentenceEncoder:
    def __init__(self):
        import os
        model_path = (FINE_TUNED_MODEL_PATH
                      if os.path.exists(FINE_TUNED_MODEL_PATH)
                      else BASE_MODEL)   # fallback to base
        self._model = SentenceTransformer(model_path)
```

The encoder is frozen at serving time. No online learning.
VRAM: 0 (runs on CPU; 22M params → ~90 MB RAM).

---

## 8. Acceptance criteria

- [ ] `train_encoder.py` completes in < 2 hours on CPU (22M param model)
- [ ] Same-section cosine ≥ 0.70 on validation set
- [ ] Cross-section cosine ≤ 0.40 on validation set
- [ ] Boundary detection F1 ≥ 0.82 on held-out fixtures
- [ ] Policy-year boundary in 3-year loss run fixture correctly detected (cosine < 0.35)
- [ ] Consecutive claim rows in same policy year NOT detected as boundary (cosine > 0.60)
- [ ] Encoder loads in < 2 s at Pipeline API startup
- [ ] Model tracked in DVC; weights not committed to Git

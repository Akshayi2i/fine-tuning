# SPEC_09 — LoRA SFT Training (Eight Task Adapters)

**Owner:** ML Engineering  
**Depends on:** SPEC_00 (task schemas), SPEC_16 (training data format)  
**Language:** Python 3.11 · transformers · peft · bitsandbytes · trl  
**Run on:** A10G or A100 (Phase 3 — offline, not serving pipeline)  
**Output:** 8 LoRA adapter checkpoints (3 task-level + 5 policy_check LOB-group variants)  
**Adapter config:** Rank-16 · alpha=32 for loss_run/quote_gen/acord_mapping; Rank-32 · alpha=64 · all linear layers for policy_check variants  

---

## 0. Reconciliation with this repository (added 2026-09-27)

This document is the Fideon pipeline's own training spec, kept here for re-checking. It is **not** the
design this repository implements, and it has deliberately not been rewritten to match — below is where the
two differ, so neither is misread as the other.

| Topic | This spec | This repository (`Documentation/Implementation MDs/SPEC_00` §13) |
|---|---|---|
| Adapters | 8: three task-level + five policy LOB-group adapters, hot-swapped | ONE unified LoRA (rank 64/alpha 128), or one per **scope** (`configs/scopes.yaml`: `unified`, `policy`, `lossrun`); per-type adapters only through the §4.2 graduation gate |
| Personal lines | `policy_check_personal_lines_v1` (Home, Personal Auto, Dwelling Fire, Watercraft) | No personal-lines scope yet; it would be a scope filtered by line of business — the split already places documents per line |
| Routing key | `layout_family` stamped on the DocumentEnvelope | `doc_type` (+ `acord_form`), and for policies the request's `lob`, which selects the line's canonical schema |
| Trainer | transformers/peft/**trl** script | **ms-swift 3** (`swift sft`), its own `Seq2SeqTrainer`; TRL is not the SFT loop |
| LoRA targets | q/v_proj (task) or all-linear (policy) | q/k/v/o and gate/up/down projections of the decoder; ViT and mergers frozen |
| Output | canonical schemas, FSM/Outlines at inference | the client's canonical LOB JSON (FieldValue envelopes, dates `MM/DD/YYYY`), vLLM structured decoding; long policies as section × page windows |
| Promotion gate | fixed `field_f1 ≥ 0.92` per adapter | absolute floors + paired-bootstrap non-inferiority vs production on the **frozen golden eval set**, run through the serving pipeline; a recorded written override is the only exception |
| Storage | `/models/base`, `/models/lora`, DVC + S3 | base in `/workspace/models`; adapters, merged and quantized models in Azure Blob; staging on the RunPod volume |
| Split | holdout set | family-level split, band by volume per type, placed per line of business; test frozen once into the golden eval set |
| Runtime | — | GPU only; every long job on the pod runs detached in tmux |

## 1. Purpose

Each LoRA adapter specialises Qwen3-VL-8B-Instruct on one insurance extraction task.
Adapters are ~32–64 MB each (~0.3% of full fine-tune cost). All eight are loaded into the
vLLM server at startup and hot-swapped per request with < 10 ms overhead.

Three adapters cover task-level extraction (loss_run, quote_gen, acord_mapping) — these
tasks have visually homogeneous layouts across LOBs, so one adapter per task is sufficient.
Policy check is different: a WC declarations page (tabular class codes, state schedules)
is visually unlike a D&O manuscript (dense prose, claims-made trigger definitions) or a
homeowners declarations page (property schedule, coverage A/B/C/D table). Attempting to
train one policy_check adapter across all LOBs produces a mediocre average or catastrophic
forgetting of minority layouts. Five LOB-group adapters, each targeting a visual layout
family, solve this without requiring one adapter per carrier.

Adapters are trained once offline (Phase 3). They are frozen in serving; updates require
a new offline training run followed by a vLLM restart.

---

## 2. Training setup

### 2.1 — Eight adapters

#### Task-level adapters (visual layout homogeneous across LOBs)

| Adapter ID | Task | Primary schema | Training examples | Key challenge |
|---|---|---|---|---|
| `loss_run_v1` | Loss run extraction | loss_run canonical schema | 500 | Multi-page claim tables, policy-year grouping |
| `quote_gen_v1` | Quote submission prep | quote_gen canonical schema | 500 | Risk attributes vary widely by SIC code |
| `acord_mapping_v1` | ACORD form mapping | acord_mapping canonical schema | 500 | 25+ form variants, field IDs must be exact |

#### Policy check adapters (split by visual layout family)

`policy_check_v1` is **removed** and replaced by five LOB-group adapters. The FSM/Outlines
layer (SPEC_08) continues to handle per-LOB JSON schema enforcement at inference — the
adapter only needs to learn the visual layout grammar of its LOB group.

The routing key for adapter selection is `layout_family`, stamped on the
DocumentEnvelope by the pipeline stage (L1 for native PDFs, L3 for scanned). SPEC_06
reads this field and selects the corresponding adapter.

| Adapter ID | Layout family | LOBs covered | Visual grammar | Training examples |
|---|---|---|---|---|
| `policy_check_casualty_fleet_v1` | `casualty_fleet` | CGL, Commercial Auto, Truckers, WC, Umbrella | Tabular class codes, state schedules, rate tables, ACORD 126/127/130 | 1500 |
| `policy_check_property_pkg_v1` | `property_pkg` | Property, BOP, Inland Marine, Flood, EQ | Multi-location schedules, coverage grids, valuation tables, ACORD 140/146/160 | 1200 |
| `policy_check_exec_specialty_v1` | `exec_specialty` | D&O, E&O, Cyber, EPLI, Crime, Bonds | Dense prose, manuscript/claims-made declarations, retro dates, sublimits | 1000 |
| `policy_check_personal_lines_v1` | `personal_lines` | Home, Personal Auto, Dwelling Fire, Watercraft | Consumer declarations pages, coverage A/B/C/D tables, ACORD 80/84/90 | 800 |
| `policy_check_benefits_misc_v1` | `benefits_misc` | Life, Group Medical/Accident, Specialty | Census tables, group schedules, benefit schedules | 600 |

**Total training data: ~5100 gold examples across all adapters (80/10/10 train/val/test split per adapter)**

#### Package policy multi-pass routing

When `ExtractionMeta.is_package=True`, the pipeline performs a three-phase multi-pass
inference run rather than a single adapter call:

1. **LOB-ID pass** — the base model (no LoRA) reads the declarations page and produces
   `confirmed_lobs: list[str]` — the authoritative list of LOBs present in the package.
   L1's `detected_lobs` is forwarded as a hint in the prompt but L3 is authoritative.
   If `detected_lobs` and `confirmed_lobs` differ, `PackagePolicyOutput.lob_detection_mismatch`
   is set for monitoring.

2. **Per-LOB adapter passes** — for each LOB in `confirmed_lobs`, the routing table
   `LOB_TO_LAYOUT_FAMILY` (SPEC_06) maps the LOB to its `layout_family`, which selects the
   correct policy_check adapter. SPEC_04 section boundary detection supplies per-LOB page
   chunks; if a LOB's chunk is missing, the declarations page chunk is used as a fallback.

   | LOB(s) | layout_family | Adapter |
   |---|---|---|
   | cgl, auto, wc, umbrella | `casualty_fleet` | `policy_check_casualty_fleet_v1` |
   | property, im, flood, eq | `property_pkg` | `policy_check_property_pkg_v1` |
   | do, eo, cyber, crime, epli | `exec_specialty` | `policy_check_exec_specialty_v1` |
   | home, personal_auto | `personal_lines` | `policy_check_personal_lines_v1` |
   | life, group | `benefits_misc` | `policy_check_benefits_misc_v1` |

3. **Assembly** — per-LOB outputs are wrapped into `PackagePolicyOutput` (SPEC_00 §5a).
   `overall_audit_passed` rolls up from all per-LOB audit results.

**No combined-LOB registry YAMLs.** The number of possible LOB combinations in package
policies is unbounded; maintaining combined YAMLs per carrier per combination is not practical.
L1 extracts only common header fields for package policies and forces routing to L3.
See SPEC_02 §2a for the package detection logic.

### 2.2 — LoRA hyperparameters

Two configs: one for the task-level adapters (visually homogeneous tasks), one for the
policy_check LOB-group adapters (high visual layout variance requires higher rank and
broader target coverage).

```python
# Task-level adapters: loss_run_v1, quote_gen_v1, acord_mapping_v1
LORA_CONFIG_TASK = {
    "r":             16,       # rank
    "lora_alpha":    32,       # scaling: effective_lr = lr * (alpha / r) = lr * 2
    "target_modules": ["q_proj", "v_proj"],   # attention projection matrices only
    "lora_dropout":  0.05,
    "bias":          "none",
    "task_type":     "CAUSAL_LM",
}

# Policy check LOB-group adapters: policy_check_*_v1
LORA_CONFIG_POLICY_CHECK = {
    "r":             32,       # higher rank — policy layout variance demands more capacity
    "lora_alpha":    64,       # keep alpha = 2×r
    "target_modules": [        # all linear layers — visual layout parsing uses FFN heavily
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    "lora_dropout":  0.05,
    "bias":          "none",
    "task_type":     "CAUSAL_LM",
}
# Trainable params with LORA_CONFIG_POLICY_CHECK: ~53M vs ~13M for LORA_CONFIG_TASK
# Adapter size: ~64 MB vs ~32 MB


TRAINING_CONFIG = {
    "num_train_epochs":        3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,    # effective batch = 8
    "learning_rate":           2e-4,
    "lr_scheduler_type":       "cosine",
    "warmup_ratio":            0.05,
    "fp16":                    False,
    "bf16":                    True,     # A10G/A100 native bfloat16
    "optim":                   "adamw_torch_fused",
    "dataloader_num_workers":  4,
    "save_strategy":           "epoch",
    "evaluation_strategy":     "epoch",
    "load_best_model_at_end":  True,
    "metric_for_best_model":   "eval_field_f1",
    "greater_is_better":       True,
}
```

---

## 3. Training script (`scripts/train_lora.py`)

```python
#!/usr/bin/env python3
"""
Usage (task-level adapter):
  python scripts/train_lora.py \
    --adapter loss_run_v1 \
    --data_dir data/sft/loss_run/ \
    --output_dir models/lora/loss_run_v1 \
    --base_model Qwen/Qwen3-VL-8B-Instruct

Usage (policy_check LOB-group adapter):
  python scripts/train_lora.py \
    --adapter policy_check_casualty_fleet_v1 \
    --data_dir data/sft/policy_check_casualty_fleet/ \
    --output_dir models/lora/policy_check_casualty_fleet_v1 \
    --base_model Qwen/Qwen3-VL-8B-Instruct
"""
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
from fideon.training.dataset import load_sft_dataset
from fideon.training.metrics import compute_field_f1

# Adapters that use the high-rank policy_check config
POLICY_CHECK_ADAPTERS = {
    "policy_check_casualty_fleet_v1",
    "policy_check_property_pkg_v1",
    "policy_check_exec_specialty_v1",
    "policy_check_personal_lines_v1",
    "policy_check_benefits_misc_v1",
}

def main(args):
    # 1. Load base model in BFloat16 (no quantization for quality)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype="bfloat16",
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Apply LoRA config — policy_check adapters use high-rank all-linear config
    if args.adapter in POLICY_CHECK_ADAPTERS:
        lora_config = LoraConfig(**LORA_CONFIG_POLICY_CHECK)
        # Expected: trainable params: ~53M | all params: ~8B | trainable%: 0.66%
    else:
        lora_config = LoraConfig(**LORA_CONFIG_TASK)
        # Expected: trainable params: ~13M | all params: ~8B | trainable%: 0.16%
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. Load training data
    train_dataset, val_dataset = load_sft_dataset(args.data_dir, args.adapter)

    # 4. Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        **TRAINING_CONFIG,
    )

    # 5. SFT Trainer (handles chat template formatting automatically)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        compute_metrics=lambda eval_pred: compute_field_f1(eval_pred, args.task),
        # Use chat template from Qwen3-VL
        dataset_text_field=None,   # we pass full messages; SFTTrainer formats them
        max_seq_length=32768,
    )

    # 6. Train
    trainer.train()

    # 7. Save adapter only (not full model)
    model.save_pretrained(args.output_dir)    # saves adapter_config.json + adapter_model.bin
    tokenizer.save_pretrained(args.output_dir)

    print(f"Adapter saved to {args.output_dir}")
    print(f"Adapter size: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} params")
```

---

## 4. Training data format  (`fideon/training/dataset.py`)

See SPEC_16 for full data format spec. Summary:

```python
def load_sft_dataset(data_dir: str, task: str):
    """
    Load JSONL files from data_dir/{task}/train.jsonl and val.jsonl.
    Each line is a training example in OpenAI messages format:
    {
      "messages": [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": "<markdown doc text or image+text>"},
        {"role": "assistant", "content": "<gold JSON output>"}
      ]
    }
    The assistant turn is the gold annotation (see SPEC_16 for annotation format).
    """
    train = load_jsonl(f"{data_dir}/{task}/train.jsonl")
    val   = load_jsonl(f"{data_dir}/{task}/val.jsonl")
    return HuggingFaceDataset(train), HuggingFaceDataset(val)
```

---

## 5. Custom evaluation metric  (`fideon/training/metrics.py`)

```python
from sklearn.metrics import f1_score
import json

def compute_field_f1(eval_pred, task: str) -> dict:
    """
    Compute field-level F1: for each field in the schema, compare
    predicted parsed value to gold parsed value.

    field_f1 = mean F1 across all required fields
    table_f1 = F1 for claim rows (loss_run) or coverage sections (policy_check)
    """
    logits, labels = eval_pred
    # Decode predictions and labels
    # (SFTTrainer provides token IDs; decode to text)
    pred_texts = tokenizer.batch_decode(logits.argmax(-1), skip_special_tokens=True)
    gold_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    field_scores = []
    table_scores = []

    for pred_text, gold_text in zip(pred_texts, gold_texts):
        try:
            pred = json.loads(pred_text)
            gold = json.loads(gold_text)
        except json.JSONDecodeError:
            field_scores.append(0.0)
            continue

        # Field-level comparison
        field_f1 = _compare_fields(pred, gold, task)
        field_scores.append(field_f1)

        # Table-level (row-by-row) comparison
        if task == "loss_run":
            table_f1 = _compare_claim_rows(pred, gold)
            table_scores.append(table_f1)

    return {
        "eval_field_f1": sum(field_scores) / len(field_scores) if field_scores else 0.0,
        "eval_table_f1": sum(table_scores) / len(table_scores) if table_scores else 0.0,
    }


def _compare_fields(pred: dict, gold: dict, task: str) -> float:
    """
    For each required field path, check if pred value == gold value (case-insensitive,
    currency-normalised). Return fraction of matching fields.
    """
    required = REQUIRED_FIELDS[task]
    matches = 0
    for path in required:
        pred_val = get_nested(pred, path)
        gold_val = get_nested(gold, path)
        if normalise_value(pred_val) == normalise_value(gold_val):
            matches += 1
    return matches / len(required) if required else 1.0


def _compare_claim_rows(pred: dict, gold: dict) -> float:
    """
    For loss run: compare claim rows by claim_number (primary key).
    For each gold claim, check if pred contains a matching claim with
    correct total_incurred (±$0.01).
    Returns fraction of gold claims correctly matched.
    """
    gold_claims = {
        c["claim_number"]["raw"]: c
        for p in gold.get("periods", [])
        for c in p.get("claims", [])
    }
    pred_claims = {
        c["claim_number"]["raw"]: c
        for p in pred.get("periods", [])
        for c in p.get("claims", [])
    }
    if not gold_claims:
        return 1.0
    matches = 0
    for key, gold_claim in gold_claims.items():
        if key in pred_claims:
            gold_total = to_decimal(gold_claim.get("total_incurred", {}).get("parsed"))
            pred_total = to_decimal(pred_claims[key].get("total_incurred", {}).get("parsed"))
            if gold_total and pred_total and abs(gold_total - pred_total) <= Decimal("0.01"):
                matches += 1
    return matches / len(gold_claims)
```

---

## 6. Promotion gate thresholds

```python
PROMOTION_GATES = {
    # Task-level adapters
    "loss_run_v1":                          {"field_f1": 0.92, "table_f1": 0.88},
    "quote_gen_v1":                         {"field_f1": 0.92, "table_f1": None},
    "acord_mapping_v1":                     {"field_f1": 0.92, "table_f1": None},
    # Policy check LOB-group adapters — text_sections are raw text, not structured fields,
    # so table_f1 is not applicable; field_f1 covers the structured coverage/limits fields.
    "policy_check_casualty_fleet_v1":       {"field_f1": 0.92, "table_f1": None},
    "policy_check_property_pkg_v1":         {"field_f1": 0.92, "table_f1": None},
    "policy_check_exec_specialty_v1":       {"field_f1": 0.92, "table_f1": None},
    "policy_check_personal_lines_v1":       {"field_f1": 0.92, "table_f1": None},
    "policy_check_benefits_misc_v1":        {"field_f1": 0.92, "table_f1": None},
}

def check_promotion(metrics: dict, adapter: str) -> bool:
    gates = PROMOTION_GATES[adapter]
    if metrics["eval_field_f1"] < gates["field_f1"]:
        return False
    if gates["table_f1"] and metrics["eval_table_f1"] < gates["table_f1"]:
        return False
    return True
```

If **any** adapter fails its promotion gate, the DAPT decision is triggered (SPEC_14).
The threshold for DAPT triggering is `field_f1 < 0.90` on any adapter (lower than
the promotion gate — DAPT is a last resort, not triggered on marginal misses).

---

## 7. Adapter versioning and deployment

```
Versioning scheme: {task}_{version}  e.g. loss_run_v1, loss_run_v2
DVC tracking: adapter checkpoints tracked in DVC, stored in S3
Git: only dvc.yaml and dvc.lock committed — never raw adapter weights

Deployment checklist:
  1. Adapter passes promotion gate (§ 6)
  2. Adapter tested on holdout set (not seen during training)
  3. P95 latency measured on RunPod with adapter loaded (must be ≤ 500 ms)
  4. VRAM measured (4 adapters must fit with base model in 24 GB)
  5. Adapter ID registered in fideon/vlm/lora_selector.py (LORA_ADAPTER_MAP)
  6. Integer LoRA ID updated in fideon/vlm/client.py (LORA_ID_MAP)
  7. vLLM server restarted with new --lora-modules entry (8 adapters total)
  8. Smoke test: one real document per adapter → correct schema output
  NOTE: For policy_check adapters, also verify layout_family is correctly stamped
  on the DocumentEnvelope by L1/L3 before the adapter selection step.
```

---

## 8. Adapter storage on RunPod

```
/models/
  base/
    Qwen3-VL-8B-Instruct/                     # base model weights (~16 GB)
  lora/
    loss_run_v1/
      adapter_config.json                      # LoRA config (rank=16, targets: q/v_proj)
      adapter_model.safetensors                # adapter weights (~32 MB)
    quote_gen_v1/ ...                          # same structure, ~32 MB
    acord_mapping_v1/ ...                      # same structure, ~32 MB
    policy_check_casualty_fleet_v1/
      adapter_config.json                      # rank=32, all linear layers
      adapter_model.safetensors                # adapter weights (~64 MB)
    policy_check_property_pkg_v1/ ...          # ~64 MB
    policy_check_exec_specialty_v1/ ...        # ~64 MB
    policy_check_personal_lines_v1/ ...        # ~64 MB
    policy_check_benefits_misc_v1/ ...         # ~64 MB
```
Total adapter footprint in VRAM: 3×32 MB + 5×64 MB = 416 MB (~0.4 GB)

---

## 9. Compute estimate

| Adapter | Config | Training examples | Epochs | A10G time | A100 time |
|---|---|---|---|---|---|
| loss_run_v1 | task | 500 | 3 | ~3 h | ~1.5 h |
| quote_gen_v1 | task | 500 | 3 | ~3 h | ~1.5 h |
| acord_mapping_v1 | task | 500 | 3 | ~3 h | ~1.5 h |
| policy_check_casualty_fleet_v1 | policy_check | 1500 | 3 | ~9 h | ~4.5 h |
| policy_check_property_pkg_v1 | policy_check | 1200 | 3 | ~7 h | ~3.5 h |
| policy_check_exec_specialty_v1 | policy_check | 1000 | 3 | ~6 h | ~3 h |
| policy_check_personal_lines_v1 | policy_check | 800 | 3 | ~5 h | ~2.5 h |
| policy_check_benefits_misc_v1 | policy_check | 600 | 3 | ~4 h | ~2 h |
| **Total** | | **~5100** | | **~40 h** | **~20 h** |

VRAM during training:
- Task-level adapters (rank-16): ~22 GB (base model BFloat16 + LoRA gradients + optimizer states) → A10G feasible
- Policy_check adapters (rank-32, all linear): ~24 GB → run on A100 (comfortable); A10G is marginal

Phase 3 budget: 30 GPU-hours (A100). Train policy_check adapters first (higher VRAM), then task-level adapters.

---

## 10. Acceptance criteria

- [ ] `train_lora.py --adapter loss_run_v1` completes without OOM on A10G 24 GB
- [ ] `train_lora.py --adapter policy_check_casualty_fleet_v1` completes without OOM on A100 40 GB
- [ ] `eval_field_f1 ≥ 0.92` and `eval_table_f1 ≥ 0.88` for loss_run_v1 on val set
- [ ] `eval_field_f1 ≥ 0.92` for each of the 5 policy_check adapters on their respective val sets
- [ ] Task-level adapter checkpoint size ≤ 40 MB (rank-16, q/v_proj only)
- [ ] Policy_check adapter checkpoint size ≤ 70 MB (rank-32, all linear layers)
- [ ] All 8 adapters can be loaded simultaneously in vLLM without exceeding 24 GB VRAM
- [ ] LoRA hot-swap between adapters adds < 10 ms per request (measured on RunPod)
- [ ] Adapter weights are tracked in DVC and not committed to Git
- [ ] `check_promotion()` correctly blocks deployment of an adapter with `field_f1 = 0.89`
- [ ] Training script selects LORA_CONFIG_POLICY_CHECK for `policy_check_*` adapters and LORA_CONFIG_TASK for others

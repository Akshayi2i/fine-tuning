# SPEC_14 — Continuous Learning Pipeline (Drift Monitor + Monthly Retraining)

**Owner:** ML Engineering  
**Depends on:** SPEC_07 (Audit Gate metrics), SPEC_09 (LoRA training), SPEC_11 (correction log)  
**Language:** Python 3.11 · Apache Airflow · Presidio  
**Files to create:** `fideon/monitoring/`, `airflow/dags/`  
**Runs:** Monthly scheduled DAG + continuous drift monitor  

---

## 1. Purpose

Model quality degrades over time as new carriers are onboarded, document formats change,
and edge cases accumulate. The continuous learning pipeline:
1. **Monitors** field_f1 and HITL rate in production (continuous)
2. **Triggers** a retraining run when quality drops below threshold
3. **Retrains** LoRA adapters on augmented data (monthly or on-trigger)
4. **Evaluates** new adapters before promotion
5. **Conditionally triggers DAPT** if standard LoRA retraining is insufficient

DAPT (Domain Adaptive Pre-Training) is a more expensive step that adjusts the base
model's token distribution. It runs only if LoRA retraining fails the promotion gate.

---

## 2. Monitoring  (`fideon/monitoring/`)

### 2.1 — Metrics collected per document

The Pipeline API Service writes metrics to Azure Monitor / Application Insights:

```python
# In fideon/monitoring/metrics_writer.py
# Called at end of every /extract request

def record_extraction_metrics(envelope: DocumentEnvelope):
    metrics = {
        "document_id":    envelope.meta.document_id,
        "tenant_id":      envelope.meta.tenant_id,
        "document_type":  envelope.meta.document_type,
        "routing_layer":  envelope.meta.routing_layer,
        "modality":       envelope.meta.modality,
        "chunked":        envelope.meta.chunked,
        "stp":            envelope.stp,
        "audit_passed":   envelope.audit_passed,
        "audit_flag_count": len(envelope.audit_flags),
        "math_fail_count":  sum(1 for f in envelope.audit_flags if "math" in f),
        "missing_field_count": sum(1 for f in envelope.audit_flags if "missing" in f),
        "processing_ms":  envelope.meta.processing_ms,
        "timestamp":      datetime.utcnow().isoformat(),
    }
    # Write to Azure Monitor custom metrics table
    telemetry_client.track_metric("extraction", metrics)
```

### 2.2 — Drift detection  (`fideon/monitoring/drift_detector.py`)

```python
DRIFT_THRESHOLDS = {
    # 7-day rolling HITL rate; trigger retraining if exceeded
    "hitl_rate_threshold":   0.10,    # 10% (production target is ≤ 5%)
    # 7-day rolling math fail rate
    "math_fail_threshold":   0.05,
    # Minimum sample size before drift detection activates
    "min_sample_size":       50,
}

class DriftDetector:
    def __init__(self, azure_monitor_client):
        self._client = azure_monitor_client

    def compute_rolling_metrics(self, window_days: int = 7) -> dict:
        """
        Query Azure Monitor for rolling metrics over the past N days.
        For task-level adapters (loss_run, quote_gen, acord_mapping): keyed by document_type.
        For policy_check adapters: keyed by (document_type, layout_family) — so that
        quality drops in one LOB group trigger retraining of only that adapter, not all five.
        """
        since = (datetime.utcnow() - timedelta(days=window_days)).isoformat()
        results = self._client.query(f"""
            customMetrics
            | where name == "extraction" and timestamp > datetime('{since}')
            | summarize
                total = count(),
                hitl_rate = avg(todouble(customDimensions["stp"] == "false")),
                math_fail_rate = avg(todouble(customDimensions["math_fail_count"] > "0"))
            by
                document_type  = tostring(customDimensions["document_type"]),
                layout_family  = tostring(customDimensions["layout_family"])
        """)
        # Build adapter-keyed metrics
        adapter_metrics = {}
        for row in results:
            doc_type = row["document_type"]
            lf = row.get("layout_family") or ""
            if doc_type == "policy_check" and lf:
                key = f"policy_check_{lf}_v1"
            else:
                key = f"{doc_type}_v1"
            adapter_metrics[key] = row
        return adapter_metrics

    def should_retrain(self, metrics: dict) -> dict[str, bool]:
        """
        Return {adapter: should_retrain} for each adapter.
        Policy_check adapters are evaluated independently — a drop in casualty_fleet
        does not trigger retraining of property_pkg or exec_specialty.
        """
        retrain = {}
        for adapter, m in metrics.items():
            if m["total"] < DRIFT_THRESHOLDS["min_sample_size"]:
                retrain[adapter] = False
                continue
            retrain[adapter] = (
                m["hitl_rate"] > DRIFT_THRESHOLDS["hitl_rate_threshold"] or
                m["math_fail_rate"] > DRIFT_THRESHOLDS["math_fail_threshold"]
            )
        return retrain
```

---

## 3. Retraining DAG  (`airflow/dags/monthly_retrain.py`)

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

with DAG(
    dag_id="fideon_monthly_retrain",
    schedule_interval="0 2 1 * *",   # 2 AM on the 1st of each month
    start_date=datetime(2026, 10, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=30)},
) as dag:

    check_drift = PythonOperator(
        task_id="check_drift",
        python_callable=check_drift_and_decide,
    )

    prepare_data = PythonOperator(
        task_id="prepare_training_data",
        python_callable=prepare_augmented_dataset,
    )

    train_adapters = PythonOperator(
        task_id="train_lora_adapters",
        python_callable=run_lora_training,
    )

    evaluate = PythonOperator(
        task_id="evaluate_adapters",
        python_callable=evaluate_new_adapters,
    )

    dapt_gate = PythonOperator(
        task_id="dapt_gate",
        python_callable=check_dapt_trigger,
    )

    promote_or_dapt = PythonOperator(
        task_id="promote_or_trigger_dapt",
        python_callable=promote_adapters_or_trigger_dapt,
    )

    check_drift >> prepare_data >> train_adapters >> evaluate >> dapt_gate >> promote_or_dapt
```

### 3.1 — Task: prepare_augmented_dataset

```python
def prepare_augmented_dataset(**context):
    """
    Build augmented training dataset:
    1. Load existing gold examples (from SPEC_16 data store)
    2. Load correction log from Azure Blob (de-identified, SPEC_11)
    3. Filter corrections from the past month
    4. Convert corrections to SFT training examples
    5. Merge with existing gold data (corrections weighted 2x)
    6. Write to data/sft/{task}/train.jsonl (overwrite)
    """
    correction_log = load_correction_log(month=context["ds"][:7])   # YYYY-MM

    new_examples = []
    for correction in correction_log:
        # Convert correction to training example
        example = correction_to_training_example(correction)
        if example:
            new_examples.append(example)

    # Task-level adapters: keyed by document_type
    task_adapters = {
        "loss_run":      "data/sft/loss_run/",
        "quote_gen":     "data/sft/quote_gen/",
        "acord_mapping": "data/sft/acord_mapping/",
    }
    # Policy_check adapters: keyed by layout_family
    policy_check_adapters = {
        "casualty_fleet": "data/sft/policy_check_casualty_fleet/",
        "property_pkg":   "data/sft/policy_check_property_pkg/",
        "exec_specialty": "data/sft/policy_check_exec_specialty/",
        "personal_lines": "data/sft/policy_check_personal_lines/",
        "benefits_misc":  "data/sft/policy_check_benefits_misc/",
    }

    # Augment task-level adapters
    for doc_type, data_dir in task_adapters.items():
        existing = load_jsonl(f"{data_dir}train.jsonl")
        corrections = [e for e in new_examples if e.get("adapter") == f"{doc_type}_v1"]
        augmented = existing + corrections * 2   # corrections weighted 2x
        random.shuffle(augmented)
        write_jsonl(f"{data_dir}train.jsonl", augmented)
        logger.info(f"{doc_type}_v1: {len(existing)} existing + {len(corrections)} corrections")

    # Augment policy_check adapters — only retrain adapters flagged by should_retrain()
    retrain_flags = context["ti"].xcom_pull(task_ids="check_drift")
    for lf, data_dir in policy_check_adapters.items():
        adapter = f"policy_check_{lf}_v1"
        if not retrain_flags.get(adapter, False):
            logger.info(f"{adapter}: no drift detected, skipping augmentation")
            continue
        existing = load_jsonl(f"{data_dir}train.jsonl")
        corrections = [e for e in new_examples
                       if e.get("adapter") == adapter
                       or (e.get("adapter", "").startswith("policy_check_") and e.get("layout_family") == lf)]
        augmented = existing + corrections * 2
        random.shuffle(augmented)
        write_jsonl(f"{data_dir}train.jsonl", augmented)
        logger.info(f"{adapter}: {len(existing)} existing + {len(corrections)} corrections")
```

### 3.2 — Task: check_dapt_trigger

```python
DAPT_TRIGGER_THRESHOLD = 0.90   # field_f1 below this → DAPT

def check_dapt_trigger(**context):
    """
    Examine evaluation results. If any adapter has field_f1 < 0.90, trigger DAPT.
    DAPT is a 2-week A100 job — only trigger when LoRA alone is insufficient.
    """
    eval_results = context["ti"].xcom_pull(task_ids="evaluate_adapters")
    need_dapt = any(
        v["field_f1"] < DAPT_TRIGGER_THRESHOLD
        for v in eval_results.values()
    )
    context["ti"].xcom_push(key="trigger_dapt", value=need_dapt)
    if need_dapt:
        failing = [k for k,v in eval_results.items() if v["field_f1"] < DAPT_TRIGGER_THRESHOLD]
        logger.warning(f"DAPT triggered for adapters: {failing}")
```

---

## 4. DAPT spec (Domain Adaptive Pre-Training)

DAPT adjusts the base model's token distribution toward P&C insurance vocabulary.
This is only triggered when LoRA SFT fails the promotion gate (field_f1 < 0.90).

```python
DAPT_CONFIG = {
    "run_on":          "A100 80GB",    # larger GPU required; A10G may OOM
    "estimated_hours": 40,             # 2-week window; run on weekends
    "base_model":      "Qwen/Qwen3-VL-8B-Instruct",
    "learning_rate":   5e-5,           # lower than SFT; preserve general capabilities
    "num_steps":       2000,
    "warmup_steps":    200,

    # Training data for DAPT: RAW TEXT (not instruction-response pairs)
    # Source: de-identified MinerU markdown from full document archive
    # NOT the same as SFT data — DAPT uses document text, not extraction annotations
    "data": {
        "source":       "data/dapt/insurance_corpus.jsonl",  # de-identified markdown docs
        "format":       "text_completion",                    # next-token prediction
        "total_tokens": "~50M",                              # from ~200K pages
    },

    # After DAPT, re-run LoRA SFT on top of new base weights
    "post_dapt_sft": True,
    "post_dapt_sft_epochs": 5,   # more epochs since base weights changed
}

# DAPT BESPOKE TOKENISER NOTE:
# DAPT may involve building a domain-specific tokeniser (BPE vocabulary builder)
# that adds insurance-specific tokens (claim IDs, ACORD form names, etc.)
# to the base vocabulary. This is SEPARATE from the Domain Sentence Encoder (SPEC_10).
# The BPE tokeniser is a one-time offline tool. It is NOT in the serving pipeline.
# If DAPT is triggered, the tokeniser update is part of the DAPT workflow:
#   1. Build bespoke BPE vocab from insurance corpus
#   2. Extend base model embedding layer with new tokens
#   3. Continue pre-training on insurance corpus
#   4. Re-run LoRA SFT on the DAPT-adjusted model
```

---

## 5. Federated weight aggregation concept

For future multi-broker deployment where each broker has a Model A instance:

```
Standard approach (current):
  All corrections aggregated centrally after de-identification.
  Single updated adapter deployed to all brokers.

Future federated approach:
  Each broker's Model A instance fine-tunes a local delta on their correction log.
  Federated Averaging (FedAvg) aggregates deltas at Fideon without seeing raw data.
  Aggregated delta applied to base model.

NOT implemented in Phase 5 — documented here for architecture awareness only.
Implementation requires: differential privacy budget tracking, secure aggregation protocol.
```

---

## 6. Retraining cadence and triggers

```
Monthly scheduled retraining:
  Day 1 of each month → check drift → if any drift → full retrain cycle

On-demand retraining triggers:
  • HITL rate > 10% for 3 consecutive days
  • Math fail rate > 5% for 3 consecutive days
  • New carrier added to registry requiring adapter update
  • 100+ corrections accumulated since last retraining

Minimum retraining interval: 2 weeks
  (Prevent thrashing if quality oscillates near threshold)

Maximum retraining interval: 3 months
  (Force retrain even if metrics look stable; prevents silent drift)
```

---

## 7. Acceptance criteria

- [ ] Drift detector correctly flags a task with HITL rate > 10% as needing retraining
- [ ] `prepare_augmented_dataset` includes corrections from Azure Blob, doubled in weight
- [ ] Monthly Airflow DAG runs on schedule and completes without manual intervention
- [ ] DAPT is NOT triggered when all adapters achieve field_f1 ≥ 0.90 after LoRA SFT
- [ ] DAPT IS triggered when any adapter has field_f1 < 0.90 after LoRA SFT
- [ ] Promoted adapters are deployed via DVC reference update (no raw weights in Git)
- [ ] All correction log data is de-identified by Presidio before entering training pipeline
- [ ] `check_dapt_trigger` correctly logs which adapters triggered DAPT

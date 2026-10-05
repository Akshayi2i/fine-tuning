# SPEC_11 — HITL Workflow & Correction Loop

**Owner:** Backend + Product  
**Depends on:** SPEC_00 (DocumentEnvelope, FideonError), SPEC_01 (hitl_queue.py), SPEC_14 (correction log → retraining)  
**Language:** Python 3.11 · Azure Queue Storage · Azure Blob · Microsoft Presidio  
**Files to create:** `fideon/hitl/`  

---

## 1. Purpose

Documents that fail the Audit Gate (SPEC_07) enter the HITL queue. A human expert reviews
the extraction result, corrects any errors, and submits a correction. Corrections are:
1. Returned immediately to the caller (corrected JSON)
2. Logged to Azure Blob (de-identified) as a training signal for monthly retraining

Target: ≤ 5% of L3 documents require HITL.

---

## 2. Module layout

```
fideon/
  hitl/
    __init__.py
    queue_consumer.py      # polls Azure Queue, dispatches to expert interface
    correction_schema.py   # CorrectionSubmission, CorrectionField types
    correction_logger.py   # de-identify and log to Azure Blob
    presidio_deidentifier.py   # PII scrubbing before storage
    tests/
      test_correction_logger.py
      test_presidio_deidentifier.py
```

---

## 3. HITL queue message schema

Written by `hitl_queue.py` (SPEC_01), consumed by the expert review interface.

```python
class HitlQueueMessage(BaseModel):
    # Routing
    document_id: str          # UUID
    tenant_id: str
    document_type: str
    queued_at: str            # ISO datetime UTC
    priority: Literal["standard", "urgent"] = "standard"
    # Failure context
    audit_flags: list[str]    # from AuditResult.flags
    stp_blocked_by: list[str] # fail-severity flags
    # Full extraction result for review
    envelope_json: str        # DocumentEnvelope.model_dump_json()
    # Original document location (for expert to view)
    document_blob_path: str   # Azure Blob path to original PDF
```

---

## 4. Queue consumer  (`queue_consumer.py`)

```python
import asyncio
from azure.storage.queue.aio import QueueClient

class HitlQueueConsumer:
    """
    Polls the Azure HITL queue and dispatches messages to the expert review service.
    Runs as a separate process from the Pipeline API Service.
    """

    def __init__(self, connection_string: str, queue_name: str, expert_api_url: str):
        self._queue_client = QueueClient.from_connection_string(connection_string, queue_name)
        self._expert_api_url = expert_api_url

    async def run_forever(self, poll_interval_s: int = 10):
        while True:
            messages = await self._queue_client.receive_messages(
                messages_per_page=5,
                visibility_timeout=300,   # 5 min; message reappears if not deleted
            )
            async for msg in messages:
                try:
                    payload = HitlQueueMessage.model_validate_json(msg.content)
                    await self._dispatch(payload)
                    await self._queue_client.delete_message(msg)
                except Exception as e:
                    logger.error(f"HITL dispatch failed for {msg.id}: {e}")
                    # Message visibility expires → automatically re-queued for retry
            await asyncio.sleep(poll_interval_s)

    async def _dispatch(self, payload: HitlQueueMessage):
        """
        POST to the expert review service (internal Fideon tool).
        Expert service provides a UI for reviewing and correcting extractions.
        """
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self._expert_api_url}/hitl/cases",
                json=payload.model_dump(),
                headers={"X-Internal-Key": INTERNAL_KEY},
            )
```

---

## 5. Expert review interface contract

The expert review service is a separate internal tool (not spec'd here) that must
implement these endpoints consumed by the HITL consumer and correction logger:

```
POST /hitl/cases
  Body: HitlQueueMessage
  Creates a review case in the expert UI; assigns to available reviewer.

GET /hitl/cases/{document_id}
  Returns case status: "pending" | "in_review" | "corrected" | "escalated"

POST /hitl/cases/{document_id}/corrections
  Body: CorrectionSubmission
  Expert submits their corrected extraction.
  → Triggers correction_logger.py
  → Returns corrected DocumentEnvelope to original caller (via webhook or polling)
```

---

## 6. Correction schema  (`correction_schema.py`)

```python
class CorrectionField(BaseModel):
    field_path: str            # dot-path e.g. "periods[0].claims[2].total_incurred.parsed"
    original_value: str | None
    corrected_value: str | None
    correction_type: Literal[
        "wrong_value",         # VLM extracted wrong value
        "missing_value",       # VLM missed the field
        "spurious_value",      # VLM hallucinated a field that doesn't exist
        "formatting",          # value correct but wrong format (date, currency)
    ]
    annotator_note: str = ""   # optional free-text comment

class CorrectionSubmission(BaseModel):
    document_id: str
    tenant_id: str
    document_type: str
    annotator_id: str          # reviewer identifier (for inter-annotator agreement tracking)
    corrected_at: str          # ISO datetime UTC
    corrections: list[CorrectionField]
    corrected_envelope_json: str  # full corrected DocumentEnvelope JSON
    # Quality metadata
    correction_time_seconds: int
    root_cause: Literal[
        "ocr_error",           # MinerU OCR was wrong
        "layout_misparse",     # table structure not correctly understood
        "ambiguous_content",   # document itself is ambiguous
        "vlm_hallucination",   # VLM generated content not in document
        "math_error",          # arithmetic error in extraction
        "other",
    ]
```

---

## 7. Correction logger  (`correction_logger.py`)

```python
class CorrectionLogger:
    """
    After expert review, log the corrected extraction to Azure Blob Storage.
    Applies Presidio de-identification before storage.
    Logged corrections feed the monthly retraining pipeline (SPEC_14).
    """

    def __init__(self, connection_string: str, container: str = "hitl-corrections"):
        self._blob_service = BlobServiceClient.from_connection_string(connection_string)
        self._container = container

    async def log(self, submission: CorrectionSubmission):
        # 1. De-identify the corrected envelope
        deidentified = presidio_deidentifier.deidentify(
            submission.corrected_envelope_json,
            document_type=submission.document_type,
        )

        # 2. Build correction record
        record = {
            "document_id": submission.document_id,
            "tenant_id": submission.tenant_id,            # kept for per-broker analytics
            "document_type": submission.document_type,
            "corrected_at": submission.corrected_at,
            "root_cause": submission.root_cause,
            "correction_count": len(submission.corrections),
            "corrections": [c.model_dump() for c in submission.corrections],
            "corrected_envelope": deidentified,            # de-identified for training
        }

        # 3. Write to Azure Blob: {tenant_id}/corrections/{year_month}/{document_id}.json
        year_month = submission.corrected_at[:7]
        blob_path = f"{submission.tenant_id}/corrections/{year_month}/{submission.document_id}.json"
        blob_client = self._blob_service.get_blob_client(self._container, blob_path)
        await blob_client.upload_blob(
            json.dumps(record, indent=2).encode(),
            overwrite=True,
        )

        logger.info(f"Correction logged: {blob_path} ({len(submission.corrections)} corrections)")
```

---

## 8. Presidio de-identification  (`presidio_deidentifier.py`)

```python
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# Fields to de-identify in extracted JSON before training use
PII_ENTITY_TYPES = [
    "PERSON",            # claimant names
    "ORG",               # insured company names (if small/identifiable)
    "PHONE_NUMBER",
    "EMAIL_ADDRESS",
    "US_SSN",
    "US_ITIN",
]

# Fields to KEEP as-is (structurally important for training)
KEEP_AS_IS = [
    "carrier",           # needed for carrier-specific pattern learning
    "claim_number",      # needed as primary key (anonymised format is fine)
    "total_incurred",    # dollar amounts are training signal
    "date_of_loss",      # dates are training signal
    "policy_number",     # anonymised format is fine (e.g. "POL-XXXX-2022")
]

def deidentify(envelope_json: str, document_type: str) -> str:
    """
    Apply Presidio NER-based de-identification to the corrected envelope JSON.
    Returns de-identified JSON string safe for training use.
    """
    # Extract text fields, run Presidio, replace with placeholders
    envelope = json.loads(envelope_json)
    _deidentify_dict(envelope, document_type)
    return json.dumps(envelope)

def _deidentify_dict(obj: dict | list, document_type: str):
    """Recursively walk the dict/list and de-identify string values."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in KEEP_AS_IS:
                continue
            if isinstance(value, str) and len(value) > 3:
                obj[key] = _scrub(value)
            elif isinstance(value, (dict, list)):
                _deidentify_dict(value, document_type)
    elif isinstance(obj, list):
        for item in obj:
            _deidentify_dict(item, document_type)

def _scrub(text: str) -> str:
    results = analyzer.analyze(text=text, entities=PII_ENTITY_TYPES, language="en")
    if not results:
        return text
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results,
        operators={
            "PERSON": OperatorConfig("replace", {"new_value": "<PERSON>"}),
            "ORG":    OperatorConfig("replace", {"new_value": "<ORG>"}),
            "DEFAULT": OperatorConfig("replace", {"new_value": "<REDACTED>"}),
        },
    )
    return anonymized.text
```

---

## 9. SLA and monitoring

| Metric | Target |
|---|---|
| HITL rate (L3 docs requiring expert review) | ≤ 5% |
| Expert review turnaround | ≤ 4 business hours |
| Queue depth alert threshold | > 50 messages |
| Correction logging latency | < 2 s after submission |
| Monthly correction log volume per tenant | Tracked; feeds retraining trigger (SPEC_14) |

---

## 10. Acceptance criteria

- [ ] A document with failing math reconciliation appears in HITL queue within 5 s of extraction
- [ ] `CorrectionSubmission` with 3 `CorrectionField` objects logs correctly to Azure Blob
- [ ] Presidio de-identification removes PERSON entities from `claimant_name.raw` field
- [ ] Dollar amounts and dates are NOT removed by de-identification
- [ ] Queue consumer deletes a message from the queue after successful dispatch
- [ ] A message that fails dispatch reappears in the queue after 5 minutes (visibility_timeout)
- [ ] Correction blob path follows `{tenant_id}/corrections/{year_month}/{document_id}.json`

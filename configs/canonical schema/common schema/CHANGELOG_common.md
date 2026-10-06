# Common model changelog

`common_model.json` is the shared envelope and building blocks that every
policy_check LOB schema composes (SPEC_21 §4.2). Changes need the approval of
the common-model owner. Adding a block or field is a minor version. Renaming or
removing one is a major version, and every LOB schema that uses it needs a
crosswalk entry.

**What goes in this file:** a field, block or coverage code belongs here when
two or more LOBs print it. Anything printed by one LOB only goes in that LOB's
own block or `coverage_codes.yaml`.

**Release steps.** A change to `common_model.json` or to any schema in
`config/canonical_schema/<doc_type>/` is committed together with:
- `python scripts/build_common_model_fields_md.py`, the field reference;
- `python tools/build_prompt_schemas.py`, the prompt schemas in
  `config/canonical_schema/prompt/` (HANDOFF_prompt_schemas.md).

CI fails while either is stale (`test_fields_reference_is_up_to_date`,
`test_g_committed_outputs_equal_a_fresh_build`).

## Unreleased — prompt schemas (HANDOFF_prompt_schemas.md, Part A)

**5 October 2026.**
- **The tool.** `tools/build_prompt_schemas.py` builds one self-contained,
  minified schema per canonical schema, following the handoff's rules 1–10.
  - Links into the common model are rewritten.
  - Only the common definitions the schema reaches are copied in, in
    common-model order; on the homeowners example that is 53 of 76.
  - `text_sections`, provenance and every `fideon:` key are removed.
  - The `lean` variant serves every adapter; `full` is for `_fallback.json`
    only.
  - A sidecar records versions, SHA-256, characters and Qwen3-VL tokens.
- **Deviation from rule 8: prompt schemas are written vertically** (indent
  2), not minified, at the developer's request, so they can be read and
  reviewed as they are. On `ocean_marine` this costs 86,129 characters and
  16,967 Qwen tokens, against 41,120 and 10,808 minified: about 6,200 more
  tokens in every system message, in training and serving alike.
- **Deviation from rule 5: `fideon:aliases` is kept** in both variants, at
  the developer's request, as reference for the model. Aliases are the labels
  carriers print for a field, and SPEC_21 §4.3 says they "feed L1 mappings
  and the training prompt". Every other `fideon:` key is still removed. On
  `ocean_marine` this keeps 91 fields' aliases (222 labels), for about 2,300
  more Qwen tokens. Whether the constrained decoder ignores the unknown
  keyword, as JSON Schema validators do, joins the handoff's open "what the
  decoder enforces" check.
- **Built so far: `ocean_marine` only.** `prompt/policy_check/ocean_marine.prompt.json`
  was rebuilt on 5 October from the SPEC_21 boat schema
  (`policy_check/ocean_marine.json` 1.0.0, on common model 1.0.0), which
  replaced the SPEC_00 v3.1.0 file the first build used. It is lean, with 53
  common definitions, 390 aliases, 67,711 characters and 15,357 Qwen tokens. It is built from today's
  `policy_check/ocean_marine.json` v3.1.0, the SPEC_00 schema still served for
  boats (sidecar `source_kind: spec00`). The SPEC_21 boat schema (Phase 2)
  replaces it, and the staleness test forces the rebuild into the same
  commit.
- **Measured on the SPEC_21 homeowners example:** `full` 44,815 characters
  (10,692 tokens), `lean` 24,118 characters (6,225 tokens). The handoff
  estimated 44,752 / 24,071 characters.
- `config/canonical_schema/prompt/` is added to `SchemaRegistry.SHARED_DIRS`,
  so it is not read as a document type.

## Unreleased — lob_schema_map.yaml

**4 October 2026 (project decision):** classic auto shares one canonical schema with personal_auto.
The L1 code `classic_auto` keeps routing "Classic Automobile" documents (`config/lob_registry.yaml`), and
its `schema_lob` is now `personal_auto` (batch code PA, prefix `PA_`, the personal_auto blocks, seeds in
`autop`), as `watercraft` reads `ocean_marine`. The personal_auto Phase 2 schema covers classic-auto
concepts (guaranteed / agreed value as `vehicles[].stated_amount`, spare parts, collector-vehicle
coverages) and its Phase 3 crosswalk migrates `policy/hagerty/classic_auto.yaml` and
`policy/essentia/classic_auto.yaml`. Until that lands, the legacy `policy_check/classic_auto.json` still
takes precedence in SchemaRegistry (a code's own file wins over its `schema_lob`); it is removed in the
same change that lands the personal_auto SPEC_21 schema (SPEC21_TEAM_PROMPT.md §7.3). No common-model
field changes.

## 1.0.0 — 3 October 2026 (field list approved by the project owner)

**lob_schema_map.yaml, 3 October 2026 (project-owner decisions):**
- Watercraft is part of the ocean_marine line, not a separate line. The `ocean_marine` entry now
  carries the personal boat line (schema `ocean_marine`, `personal_lines`, batch code OM, prefix `OM_`,
  seed folder `boat`); the L1 code `watercraft` still routes to the same schema. The open item
  "ocean_marine (L1 code) has no schema" is closed.
- The seed PDFs under `synth_core/fideon/synth_core/data/source data/` are the de-identified seeds
  (SPEC_21 D2).
- The common-model field list is approved.

This is the first version for LOB developers to build on, delivered as the
Common Model Developer Brief asks. LOB developers start their step 3 once the
project owner approves the field list and the open items in
`lob_schema_map.yaml`.

**Files:**
- `common_model.json`
- `common_model.fields.md`, generated by
  `scripts/build_common_model_fields_md.py` from the JSON and the map
- `lob_schema_map.yaml`: one entry per L1 code (48), with open items
- `examples/personal_lines_example.schema.json`
- `examples/homeowners_minimal.json`
- `test_report.txt`: the `-v` run of `fideon/schemas/tests/test_common_model.py`

**HANDOFF 1.0.0 fixes (3 October 2026):**

- **Item 1 (D-A, D-B):** `lob_registry.yaml` is replaced by
  `lob_schema_map.yaml`, keyed by the 48 codes of `config/lob_registry.yaml`
  in that file's order.
  - `watercraft` reads against the `ocean_marine` schema.
  - The L1 code `ocean_marine` (commercial hulls) has no schema yet.
  - 20 codes have no layout family.
  - SPEC_06 `group` is an open item.
  - The columns `source_pdfs` and `holdout_carrier` from the replaced file
    are kept.
  - `fideon/schemas/schema_registry.py` reads the map's `schema_lob` column.
    Without this, a code whose schema has another name (`watercraft`) fell
    back to `_fallback`, because no `watercraft.json` exists.
- **Item 2:** the test file reports into `test_report.txt`. The "every date
  leaf rejects 2025-06-04" check now swaps every date in the example one at
  a time; the example gained a print date and a billing due date for it.
- **Item 3:** `field_inventory.csv` was built, and its gap list approved
  (3 October 2026).
  - **Source:** 26,175 rows. 5,077 come from the 113 personal-lines PDFs in
    `synth_core/fideon/synth_core/data/source data/`: every "Label: value" and every table column
    header, with value type, page and carrier. 21,098 come from the
    captions and `canonical_path` values of the 292 policy registry YAMLs.
  - **Deviations from the HANDOFF prompt:**
    - The seed PDFs were read in place from their folder, not copied to
      `data/seeds/`; that folder holds each seed's gold, varmap and review.
    - A `canonical_path` column was added.
    - Labels that carry a printed date were dropped: a committed file
      derived from the seeds holds labels only, never policy data.
  - **Value types corrected (3 October 2026):**
    - **The problem.** The first build read a label's value only from the
      same text line, so 47% of PDF rows had `value_type` "empty". That
      included nearly every "Named Insured", "Insured Name", "Agency" and
      "Effective Date".
    - **Why.** On declarations pages the value is usually a separate text
      block, either to the right of the label or under it as a heading.
    - **The fix.** Values are now found by position: to the right on the
      same visual line (the first value word may sit up to 150 pt away, in
      a column), else in the cell directly below the label.
    - **Result.** Empty values fell to 8% (374 of 4,919 PDF rows), mostly
      genuinely blank form fields and headings.
    - **Effect on the gap list.** Re-running it added 4 candidates, all
      policy-wording fragments; the approved model changes are unaffected.
  - **Person names removed (3 October 2026).** A driver list printed as
    "<name>: Married, Male Driver, age 64, …" made four people's names into
    labels. A row whose value describes a person (sex, marital status,
    driver, age, years driving) is now dropped; 5 rows went. The names were
    in the versions pushed earlier (`d866011`, `c618e6f`), and Git history
    still holds them.
  - **Gap rule:** 2 or more LOBs and 2 or more carriers. Duplicate carrier
    spellings were merged ("All State" and `allstate`), and
    `ocean_marine`/`watercraft` counted as one boat line. 241 candidate gaps
    became 145 after the approved changes; the rest are listed below.
  - **New fields:**
    - `policy.policy_type`: 14 LOBs, 48 carriers.
    - `policy.original_inception_date`: 3 LOBs, 3 carriers.
    - `Vehicle.engine_displacement`: 3 LOBs, 2 carriers.
    - `producer.web_address`: 2 LOBs, 2 carriers.
  - **Garaging:** option (a). A vehicle's printed garaging ZIP or state is a
    `locations[]` entry, which `vehicles[].garaging_location_ref` points to.
    The labels "Garaging Address", "Garaging ZIP Code", "Garaging State" and
    "Garage" are on `locations[].address`.
  - **About 90 printed labels added to existing fields:** policy number and
    dates, rating state, transaction, carrier, producer, insured, premium
    total and change, fees, discounts, payment method, vehicle, driver,
    location, building, valuation, forms. Also added to the blocks:
    coverages, vehicles, interested parties, forms. And to the shared codes
    `X_VEHICLE_LIABILITY`, `X_MED_PAY` and `X_PIP`.
  - **Not added:**
    - Printed by one carrier only: Number of Times Renewed, Tier, Interest
      Type, Pay in Full. "Policyholder Since" was kept only as a label.
    - Boat-only, so they belong in the watercraft LOB block: Total
      Horsepower, Propulsion Type.
    - "Basement Type": its printed values are not yes/no.
    - Policy-condition headings: go to `text_sections`.
    - Generic words and letter text.
    - Coverage-specific limits and deductibles: go to each LOB's coverage
      codes.
- **Item 4 (reversed):** the 419 source PDFs were moved from
  `synth_core/fideon/synth_core/data/source data/` to
  `data/raw/source_pdfs/`. On 3 October 2026 they were moved back to
  `synth_core/fideon/synth_core/data/source data/` at the developer's
  request, so item 4 is open again.
  - The PDFs are not in Git there either: the root `.gitignore` ignores
    every `*.pdf`.
  - **DVC tracking was not done.** DVC is not installed here.
  - **Correction (3 October 2026):** the HANDOFF called these PDFs "not
    de-identified" and "not seeds". The developer confirmed they are the
    seed data, which SPEC_21 decision D2 states arrive de-identified, so
    they can be tracked through DVC as `SYNTHETIC_DATA_PROCESS.md`
    describes.
  - None of them was ever committed, so there is no Git history to clean.
- **Item 5:** the brief, the review and the HANDOFF moved to
  `Documentation/Synthetic data generation docs/`.
- **Items 6–7 (not done):** wait for `synth_core_0.1.1.patch`, which is not in
  the repository.

**Generic printed labels removed (3 October 2026):** L1 matches aliases against
the captions printed on the page, so a one-word label claims every value
printed under it.

| Field | Removed | Why |
|---|---|---|
| `countersignature.date` | "Date" | Replaced by "Countersignature Date" and "Date Countersigned". |
| `carrier.address` | "Mailing Address" | It is the insured's address, and the label already belongs to `named_insured.mailing_address`. |
| `carrier.claims_email` | "Email" | Would catch the insured's or the agent's email. |
| `carrier.claims_address` | "Claims" | Heads claim history and claim instructions too. |
| `carrier.name` | "Company" | On umbrella policies it heads the underlying insurer column. |

A test now rejects any alias that is a single generic word ("Date", "Name",
"Email", "Claims", "Company", "Number", "Amount" and a few more).

**Content:**
- The `FieldValue` leaf and typed variants of it.
- Every envelope block (SPEC_21 §4.2.1).
- Every building block in the §4.2.2 tables, including those no
  personal-lines LOB uses.
- `lob_parts`, `additional_fields` and `text_sections`.
- 19 shared `X_` coverage codes.
- Four default audit rules.

The fields and codes chosen by the two-LOB rule are listed under the 0.2.0
and 0.3.0 drafts below.

**Changed from the 0.3.0 draft to meet the brief:**

| Brief | Change |
|---|---|
| Rule 1 | `BooleanValue` renamed `BoolValue`. |
| Rule 7 | Address `line_1` and `line_2` renamed `street` and `street_2`. synth_core reads `named_insured.mailing_address.street`. |
| Rule 4 | `attachment.retained_limit` renamed `retained_amount`; the printed label "Retained Limit" stays as an alias. It is an amount the insured retains, not a limit of the policy. |
| Rule 6 | `provenance` closed, listing the keys `synth_core/engine.py` writes plus `provenance`, the free-text note on a seed gold. |
| Rule 8 | `fideon:tier` is now only `core` or `overflow`. `text_sections` is marked by `fideon:fsm_exclude` alone. |
| Deliverables | Every field has a meaning (`description`). The build fails if one does not. |

**L1 registry check:** all 293 policy registry YAMLs were read for the table
columns they map (vehicle, endorsement and schedule columns). 44 column fields
are used by 2 or more LOBs; 35 already had a home.
- Two needed only a printed label: "Base Form" (on
  `lob_parts[].coverage_part_form`) and "Drivers" (on `drivers[].name`).
- "Limit / Amount" on a forms schedule is not a field: a form that carries a
  limit is recorded as a coverage whose `form_refs` names the form, so limits
  stay in `coverages[].limits[]`.
- The endorsement obligation columns (duty, mandatory, timeframe) describe
  wording inside an endorsement and belong in `text_sections`.
- The rest are printed by one carrier only.

**Changes from the SPEC_21 review (Lakshman, 3 October 2026):**

- **#19** — `part` added to every risk unit (locations, buildings, vehicles,
  drivers, watercraft, scheduled items). Coverages, forms and premium items
  already had it. Without it, a package policy cannot be split per LOB.
- **#6** — "at least one coverage with a limit" became the default audit rule
  `len(coverages[].limits[]) >= 1`; a mandatory-field list cannot express it.
- **#12** — `text_sections` is the only open-ended map, and it is
  FSM-excluded. `additional_fields` stays in the training target (SPEC_21
  §10.2): it is a list of label and value pairs, not a map.
- **`lob_registry.yaml`** (since replaced by `lob_schema_map.yaml`) gains
  three fields per LOB:
  - `l1_code`: the pipeline's LOB code, which names the schema file;
  - `source_pdfs`: the folder, carriers and files for the LOB;
  - `holdout_carrier`: TBD (#22), plus open items for the LOBs that cannot
    hold a carrier out.
- **Loader (`fideon/schemas/schema_registry.py`):**
  - A SPEC_21 LOB schema is no longer merged with the old `_common.json`
    (#17). The common-model definitions it uses are inlined into it, so every
    reader gets one self-contained schema.
  - Loading is refused when the common model is missing, when its version
    does not match, when a `$ref` points to an unknown file, or when a LOB
    definition shadows a common one.
  - The FSM generation schema now drops `fideon:fsm_exclude` properties (#12).
  - Mandatory fields written as leaf paths and selectors (`carrier.name`)
    are now resolved by the AuditGate and by L1's coverage rate. Before, the
    first SPEC_21 schema would have failed every document.

**Known departures from the brief:**
- **Two limit fields outside `coverages`.** `underlying_insurance[].limits[]`
  stays because SPEC_21 §4.2.2 puts it there: the limits belong to the
  underlying policy, not this one. `claims_made_terms.defense_within_limits`
  is a yes/no flag, not an amount. The test lists both as named exceptions.
- **The examples reference `../common_model.json`, not
  `../common/common_model.json`.** They live in `common/examples/`, and
  references resolve by file location. A real LOB schema in
  `policy_check/<lob>.json` uses `../common/common_model.json`.
- **The test lives in `fideon/schemas/tests/test_common_model.py`, not a
  top-level `tests/`.** CI runs `pytest fideon`, so a test anywhere else would
  never run.
- **The seed folder the brief names (`Synthetic Data Generator
  Test/Input/`) is not in the repository.** `field_inventory.csv` was built
  from the source PDFs in `synth_core/fideon/synth_core/data/source data/` and the registry YAMLs
  instead (see HANDOFF item 3 above).

## 0.3.0 — 3 October 2026 (draft, not yet approved)

This version comes from a gap scan. Every printed label (text followed by
":") was collected from the first 4 pages of all 419 source PDFs. Labels used
in 2 or more LOB folders, by 2 or more carriers, were then checked against the
field names and aliases already in the model.

**New fields:**

| Where | Added | LOBs seen in |
|---|---|---|
| `document` | `mailing_date` (starts the notice period on cancellation and nonrenewal notices) | autop, home, flood, boppr_cgl |
| `carrier` | `fax`, `claims_email`, `claims_address` (where notices are sent) | wc, pl, eo, epli, cumbr, home, autop and others |
| `producer` | `fax`, `contract_number` | eo, pl, home, bopgl, inmrc |
| `named_insured` | `fein`, `business_description` | bopgl, cgl_prop, wc, dfire |
| `policy` | `certificate_number` | cgl, cgl_prop |
| locations | `fire_district`, `insured_interest` (owner, deeded owner, tenant, LLC member) | home, dfire, boppr_cgl, cgl_prop, bopgl_boppr |
| buildings | `basement` | home, flood, cgl_prop |
| interested parties | `is_payor` (the mortgagee pays the premium) | dfire, flood, home |

**Aliases:** carriers print some existing fields under other labels. These
labels were added as aliases, not as new fields:
- "Inception Date": effective date
- "Zone" / "Rating Zone": territory
- "Protection": protection class
- "Underwritten by", "Insuring Company", "Issued by": carrier name
- "Home Office", "Administrative Office": carrier address
- "Amended Date": transaction date
- "Process Date", "Printed": print date
- "Policy Issued": issue date
- "Loan" / "Loan #": loan number
- "Prior or Pending Date": pending/prior litigation date
- "Square Footage": area in square feet
- "Attn": address line 2
- "Endorsement No": endorsement number
- "Representative": countersignature name
- surplus lines tax, stamping fee and fire fee labels: taxes and fees
- protective-device labels (smoke alarm, extinguisher, opening protection):
  protective devices

**Left out:**
- "Account ID": printed only by Coterie and its carrier Benchmark, so it is
  one company's form.
- About 30 labels that are headings, letter text or marketing copy.

**Limitation:** this scan sees only `Label: value` text. Labels used only as
table column headers are not covered.

## 0.2.0 — 3 October 2026 (draft, not yet approved)

Additions chosen by the rule above, from a scan of the 419 source PDFs
(first 4–6 pages of each).

**New fields:**

| Where | Added | LOBs seen in |
|---|---|---|
| `document` | `transaction_date`, `transaction_effective_date`, `transaction_reason`, `endorsement_number` | home, dfire, pumbr, autop, recv, flood, cyber, bopgl, boppr, cgl_prop, inmrc, health |
| `carrier` | `claims_phone` | autop, home, pumbr, recv, autob, cgl, cgl_prop, bopgl, wc |
| `named_insured` and additional named insureds | `date_of_birth`, `gender`, `marital_status`, `occupation` | home, dfire, pumbr, autop |
| `policy` | `rating_state` | autop, cycle, recv |
| `policy` | `audit_period`, `subject_to_audit` | bopgl, pl, cgl, cgl_prop, prop, autob |
| `premium` | `minimum` (separate from `minimum_earned`), `change` (negative for a return premium) | wc, cgl, inmrc, bldrk, other; most lines print a premium change |
| fees, taxes and charges | `fully_earned` | flood, cgl, do, eo, cumbr |
| `billing` | `bill_type`, `payment_method`, `account_number`, `amount_due`, `due_date` | autop, home, pumbr, cycle, recv, autob, prop, cgl_prop, bldrk |
| drivers | `age`, `gender`, `marital_status` | autop, cycle, recv, boat |
| vehicles | `annual_mileage` | autop, cycle, recv |
| vehicles | `commute_miles_one_way` | cycle, recv |
| vehicles | `attached_to_ref`, so a trailer is a vehicle linked to what tows it | boat, recv, cycle, autob |
| buildings | `heating_type`, `heating_fuel`, `heating_year_updated` | home, dfire |
| coverages | `covered_auto_symbols` | autob, package auto parts |
| new optional block | `countersignature` | many lines |

**Shared coverage codes:** `fideon:shared_coverage_codes` lists 19 `X_`
codes, each seen in 2 to 17 LOB folders: vehicle liability, medical payments,
personal liability, UM, UIM, PIP, collision, comprehensive, towing, rental,
loss of use, ordinance or law, water backup, equipment breakdown, identity
theft, cyber, EPL, hired and non-owned auto, and terrorism. Terrorism is a
coverage, so whether it was accepted or rejected goes in `included`; it does
not need a block of its own.

**New rules:**
- An empty block `{}` is rejected.
- A value that was not printed (`raw` null) must have `parsed` null too.
- A sublimit needs a `description`.
- A percentage limit needs a `basis_coverage_code`.
- A flat deductible needs an `amount`; a percentage deductible needs a
  `percentage`.
- New typed values: `StateValue` (USPS two-letter code), `PostalCodeValue`
  (5 digits or ZIP+4), `NaicValue` (5 digits) and `YearValue` (1900–2100).
- `lob_parts` must have at least one part.
- `text_sections` keys must have the form `form|heading|sub-path`.

**Default audit rules:** `fideon:default_audit_rules` are cross-field rules
that every LOB copies into its own `fideon:audit_rules`. A rule is skipped
when the values it needs are absent.
- Effective date is before expiration date.
- The term in months matches the policy dates.
- The change effective date falls within the policy period.
- The retroactive date is on or before the effective date.

**Deliberately left out:**
- **Flood zone:** only flood prints it, so it goes in the flood LOB block.
- **Date licensed and distance to coast:** no source PDF prints them.
- **Employee benefits liability:** printed under one LOB only.
- **Old schemas as evidence:** 13 of the old per-LOB schemas share one
  generated block, so a field appearing in many of them is not evidence that
  many LOBs print it.

**Single-carrier fields:** `heating_*` (Dryden Mutual) and
`commute_miles_one_way` (Allstate) meet the two-LOB rule, but each comes from
one carrier's form. Confirm them with a second carrier before approval.

## 0.1.0 — 3 October 2026 (draft, not yet approved)

First draft, written from SPEC_21 §4.2–4.6. It still needs to be checked
against a sample of seed PDFs, and the owner has not signed it off.

- **Envelope:** `document`, `carrier`, `producer`, `named_insured`, `policy`,
  `lob_parts`, `coverages`, `deductibles`, `interested_parties`, `premium`,
  `billing`, `forms_and_endorsements`, `additional_fields`, `text_sections`.
- **Building blocks:** `locations`, `buildings`, `vehicles`, `drivers`,
  `watercraft`, `scheduled_items`, `rating_exposures`, `rating_modifiers`,
  `claims_made_terms`, `insuring_agreements`, `underlying_insurance`,
  `attachment`, `bond`, `benefit_plans`.
- **Typed values:** typed versions of `FieldValue` (`DateValue`, `MoneyValue`,
  `PercentValue`, `NumberValue`, `BooleanValue`, and versions limited to a
  fixed list of values) restrict `parsed`:
  - `DateValue.parsed` must be MM/DD/YYYY (D1). It is null when the printed
    text is not a date, e.g. a retroactive date printed "Full Prior Acts".
  - `MoneyValue.parsed` must be a number. It is null when the printed text is
    not an amount ("Included", "Not Covered").
- **Structure:**
  - Every block rejects fields it does not define, so a LOB schema cannot add
    fields to a shared block.
  - Every risk unit requires a `unit_id`.
  - A printed value (`raw` set) must name at least one page (`page_ref`).
- **Not included (deliberately):**
  - Location boxes: these stay in `varmap.json` and `ocr.json`, not in
    `FieldValue`.
  - `printed_lines` (removed by SPEC_21 §4.2.6).
  - From the old `_common.json`: terrorism, loss history, state notices, claim
    reporting and signature blocks. Their values go to `additional_fields` and
    their wording to `text_sections`. Any of them that turns up across carriers
    is a candidate to add back as a core field.

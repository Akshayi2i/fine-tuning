# Change requests - personal_umbrella (to the common-model owner)

Raised while building the personal_umbrella SPEC_21 schema 1.0.0 (05 October 2026). Nothing here is applied: the schema composes `common_model.json` 1.0.0 unchanged, and each concept below stays in `additional_fields` or in the LOB coverage code named until a request is accepted. Labels only; no seed values.

**CR-PU-01. Required minimum underlying limits.**
Both carriers print a table of the minimum limits the insured must keep on underlying policies (NYCM 'Required Minimum Underlying Limits of Insurance', Plymouth 'Required Minimum Liability Limits for Underlying Policies'). No block holds it. Same as Khushi's CR-PU-3 (option A: a `required_underlying_limits[]` element of coverage code, printed name and `Limit` entries). Until decided: `additional_fields` + `text_sections`.

**CR-PU-02. Residences and exposure counts.**
Both carriers print residences (NYCM 'Residence Summary', Plymouth 'Insured Property Location') and exposure counts (NYCM 'Included Risk(s)', Plymouth 'Exposures'). Same as Khushi's CR-PU-4: add `locations` and `rating_exposures` to personal_umbrella `blocks_used`. This schema does not compose them yet (the seed golds keep both in `additional_fields`).

**CR-PU-03. Labels as aliases.**
'Insurance Provided By', 'Policy Issued by' (carrier.name), 'Term Length' (policy.term_months), 'Basic Form' (lob_parts[].coverage_part_form) are printed by the umbrella seeds and are not common-model aliases (Khushi's CR-PU-5). They are in this schema's `fideon:aliases`.

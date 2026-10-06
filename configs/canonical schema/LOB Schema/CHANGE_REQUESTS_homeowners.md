# Change requests - homeowners (to the common-model owner)

Raised while building the homeowners SPEC_21 schema 1.0.0 (05 October 2026). Nothing here is applied: the schema composes `common_model.json` 1.0.0 unchanged, and each concept below stays in `additional_fields` or in the LOB coverage code named until a request is accepted. Labels only; no seed values.

**CR-HO-01. Dwelling characteristics printed by two or more homeowners carriers with no core field.**
Travelers, NYCM and Madison Mutual print basement type / foundation type ('Basement Construction', 'FOUNDATION TYPE'), number of families and residence employees, swimming pool, garage, exterior wall and roof age; Travelers and NYCM print a heating source. `Building` has `basement` (BoolValue), `heating_type` and `heating_fuel` only. Proposed for `Building`: `foundation_type`, `number_of_families` (or use `number_of_units`: confirm), `roof_year`, `swimming_pool` (BoolValue), `garage_type`. Until accepted they stay in `additional_fields` (crosswalk entries `overflow`).

**CR-HO-02. Wind / hail and named-storm deductibles as percentages of Coverage A.**
Plymouth and Travelers print a wind or named-storm deductible as a percentage of the dwelling limit. `Deductible` has `percentage` and `percentage_basis`, which covers it; confirm `deductible_type: percentage` with `peril` 'Wind/Hail' is the intended shape (no change asked if yes).

**CR-HO-03. `premium.items[]` subtotals and fire fees.**
Midstate prints a fire fee, Leatherstocking a basic premium and Travelers a coverage premium and a premium change: the first two are `taxes_fees[]` and `items[]`, but no field says an item is a subtotal (see the dwelling fire request). Same proposal: `PremiumItem.subtotal` (BoolValue).

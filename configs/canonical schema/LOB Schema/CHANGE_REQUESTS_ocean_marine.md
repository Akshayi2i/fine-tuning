# Change requests - ocean_marine (to the common-model owner)

Raised while building the ocean_marine SPEC_21 schema 1.0.0 (05 October 2026). Nothing here is applied: the schema composes `common_model.json` 1.0.0 unchanged, and each concept below stays in `additional_fields` or in the LOB coverage code named until a request is accepted. Labels only; no seed values.

**CR-OM-01. Boat trailer.**
Progressive ('Trailer information Year ... Make ...'), Travelers ('Trailer Details') and Markel print the boat trailer; the line composes no `vehicles` block. Proposed: allow `vehicles` for ocean_marine with `body_type` 'Trailer' and `attached_to_ref` to the watercraft (the field exists), i.e. add `vehicles` to the ocean_marine `blocks_used`. Until then the trailer stays in `additional_fields`.

**CR-OM-02. Watercraft fields.**
Printed per boat by Progressive, Travelers and Markel: registration number, propulsion / engine type, number of motors, total horsepower without a per-motor breakdown, hull material, mooring location / ZIP. Proposed for `Watercraft`: `registration_number`, `propulsion_type`, `number_of_motors`, `total_horsepower`, `hull_material`, `mooring_location`.

**CR-OM-03. Operator fields.**
Travelers prints years of boating experience, licence type and boating-course membership per operator. Proposed for `Driver`: `years_experience`, `boating_course` (BoolValue).

**CR-OM-04. Shared-code aliases.**
X_VEHICLE_LIABILITY: 'Liability To Others', 'Watercraft Liability', 'Liability (Protection and Indemnity) Coverage'; X_UM: 'Uninsured Boater Coverage', 'Uninsured Watercraft'; X_MED_PAY: 'Medical Payments Coverage'; X_TOWING: 'Emergency Towing and Assist', 'Commercial Towing and Assistance Coverage'.

**CR-OM-05. Shared codes for disappearing deductible and replacement-cost personal effects.**
Disappearing / diminishing deductible: boat (Progressive, Travelers) and homeowners (Madison Mutual, Travelers). Replacement cost personal effects: boat (Progressive) and RV (Progressive). Proposed: `X_DISAPPEARING_DEDUCTIBLE`, `X_PERSONAL_EFFECTS_REPLACEMENT_COST`.

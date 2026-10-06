# Change requests - personal_auto (to the common-model owner)

Raised while building the personal_auto SPEC_21 schema 1.0.0 (05 October 2026). Nothing here is applied: the schema composes `common_model.json` 1.0.0 unchanged, and each concept below stays in `additional_fields` or in the LOB coverage code named until a request is accepted. Labels only; no seed values.

**CR-PA-01. Split X_PIP: optional basic economic loss and additional PIP as shared codes.**
`X_PIP` lists 'OBEL', 'Optional Basic Economic Loss' and 'Additional PIP' as aliases of personal injury protection. New York policies print basic PIP, OBEL and additional PIP as three coverages, each with its own limits and premium (Hagerty, NYCM, Plymouth, Progressive, Travelers personal auto; NYCM and Progressive motorcycle; Progressive and American Modern recreational vehicle). One code for three coverages of one policy breaks code alignment (SPEC_21 4.2.4). Proposed: shared `X_OBEL` and `X_ADDITIONAL_PIP`, and those aliases removed from `X_PIP`. Until accepted the line uses `PA_OPTIONAL_BASIC_ECONOMIC_LOSS` and `PA_ADDITIONAL_PIP`.

**CR-PA-02. Shared code for supplemental spousal liability.**
Printed by personal auto (Plymouth, Progressive, Travelers, NYCM, Mercury), motorcycle and recreational vehicle seeds (New York requires the offer). Proposed: `X_SUPPLEMENTAL_SPOUSAL_LIABILITY` (same as CR-RV-03). Until accepted: `PA_SUPPLEMENTAL_SPOUSAL_LIABILITY`.

**CR-PA-03. Shared code for trip interruption.**
Printed by personal auto (Travelers 'TRAVEL - TRIP INTERRUPTION', Mercury 'TRIP INTERRUPTION') and recreational vehicle (Progressive 'Trip Interruption'). Proposed: `X_TRIP_INTERRUPTION`.

**CR-PA-04. Pay-in-full amount on `premium`.**
Progressive prints 'Total 6 month policy premium if paid in full' / 'Discount if paid in full' on personal auto, motorcycle, boat and RV; the pay-in-full total differs from `premium.total`. Same as CR-RV-02: `premium.pay_in_full_total` (MoneyValue).

**CR-PA-05. Aliases for X_VEHICLE_LIABILITY and X_TOWING.**
Progressive prints 'Liability To Others' as the liability coverage name on auto, motorcycle, boat and RV; NYCM prints 'BODILY INJURY SPLIT LIMITS'; 'Towing and Labor Limit' (Hagerty). Add as aliases.

**CR-PA-06. `applies_to` on `Charge` (per-vehicle discounts and surcharges).**
NYCM, Plymouth, Progressive and Travelers print discounts per vehicle. Same as CR-RV-01.

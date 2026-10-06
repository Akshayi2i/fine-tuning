# Change requests - recreational_vehicle (to the common-model owner)

Raised while building the recreational_vehicle SPEC_21 schema 1.0.0 (05 October 2026). Nothing here is applied: the schema composes `common_model.json` 1.0.0 unchanged, and each concept below stays in `additional_fields` or in the LOB coverage code named until a request is accepted. Labels only; no seed values.

**CR-RV-01. Same as Sakshi's CR-RV-01 .. CR-RV-04 and CR-RV-07 (not repeated in full).**
`applies_to` on `Charge`; `premium.pay_in_full_total`; shared codes for supplemental spousal liability and guest/passenger liability; a `tier` rating-modifier type. All are printed by the RV seeds used here (Progressive, All State, American Modern, Foremost).

**CR-RV-02. Pedestrian PIP and the PIP split.**
Progressive prints 'Mandatory Pedestrian Personal Injury Protection' on snowmobile policies, as motorcycle does; with CR-PA-01 (X_OBEL, X_ADDITIONAL_PIP) and CR-MC-01 (X_PEDESTRIAN_PIP) the RV codes `RV_PEDESTRIAN_PIP`, `RV_OPTIONAL_BASIC_ECONOMIC_LOSS`, `RV_ADDITIONAL_PIP` become shared codes.

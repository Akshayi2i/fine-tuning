# Change requests - dwelling_fire (to the common-model owner)

Raised while building the dwelling_fire SPEC_21 schema 1.0.0 (05 October 2026). Nothing here is applied: the schema composes `common_model.json` 1.0.0 unchanged, and each concept below stays in `additional_fields` or in the LOB coverage code named until a request is accepted. Labels only; no seed values.

**CR-DF-01. `premium.items[]` for a location total.**
Dryden Mutual and Leatherstocking print a total annual premium per location ('Total Annual Premium for Location # 1', 'TOTAL LOCATION PREMIUM'). It is a premium item with `unit_type: location` and `applies_to`, which the model allows, but `PremiumItem` has no way to say it is a subtotal of other items, so the premium rule cannot use it. Proposed: a `subtotal` BoolValue on `PremiumItem`. Until then the seeds keep it in `additional_fields`.

**CR-DF-02. Building protective devices and alarms.**
Dryden Mutual ('Protective Devices', 'Type of Device', 'Burglar Alarm', 'Fire Alarm', 'Sprinklers') and Leatherstocking ('LOCAL FIRE ALARM') print alarms per building; `buildings[].protective_devices[]` and `sprinklered` exist, but the seeds also print a device *type* and a credit; no change is asked for the fields, only confirmation that the credit belongs in `premium.discounts[]` and the device in `buildings[].protective_devices[]`.

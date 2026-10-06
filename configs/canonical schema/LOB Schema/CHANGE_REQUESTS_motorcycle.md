# Change requests - motorcycle (to the common-model owner)

Raised while building the motorcycle SPEC_21 schema 1.0.0 (05 October 2026). Nothing here is applied: the schema composes `common_model.json` 1.0.0 unchanged, and each concept below stays in `additional_fields` or in the LOB coverage code named until a request is accepted. Labels only; no seed values.

**CR-MC-01. Shared codes printed by motorcycle and recreational vehicle.**
Guest / passenger liability (MC: RT Specialty 'Guest Liability', All State 'PASSENGER LIABILITY'; RV: All State, American Modern), accessories / optional equipment (MC: all three carriers; RV: Progressive, American Modern), wrongful death (MC: All State; RV: All State), pedestrian PIP (MC: NYCM/Progressive; RV: Progressive). Proposed: `X_GUEST_PASSENGER_LIABILITY` (as CR-RV-04), `X_ACCESSORIES`, `X_WRONGFUL_DEATH`, `X_PEDESTRIAN_PIP`. Supplemental spousal liability and OBEL: see CR-PA-01/02.

**CR-MC-02. Policy tier as a rating modifier type.**
Progressive and All State print a policy tier on motorcycle and RV policies. Same as CR-RV-07.

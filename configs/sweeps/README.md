# Hyperparameter sweeps - AUTHORED NOW, RUN LATER

These define the bounded 3-phase protocol from arch 11a so the method is settled
before anyone is tempted to improvise. **Do not run them this cycle.**

A 9-12 run sweep against 25-30 documents per type mostly measures noise - arch 8
itself calls pilot metrics directional. Arch 11a scopes the sweep to "before the
first **production** run", not before the pilot.

Sequence: pass the SPEC_15 pilot, then sweep, then train at production scale.

Every sweep run still writes a full run manifest with `is_sweep_run: true`
(SPEC_02), so sweep runs are first-class registry entries rather than untracked
side experiments. Total budget: ~9-12 runs before the first production run.

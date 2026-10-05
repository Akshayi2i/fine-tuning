"""Orchestration — the operator command surface (IMPL-13).

Three commands plus one umbrella, so a cycle is run by a person who knows the
version tag and nothing else about paths, pods, or stage wiring:

``finetune`` (stages 1→7) · ``package`` (8→9) · ``extract`` (§17) · ``all``.

``all`` is ``finetune`` + ``package`` and **never** extraction — producing a model
and using one are different concerns, and folding them together would make every
build wait on a test run against a model that may not exist yet.

This package runs on cheap CPU infrastructure **outside** RunPod (arch §14). Only
GPU-bound work is dispatched to a pod.
"""

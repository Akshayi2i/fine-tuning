"""Each answer capped at its section task's reserved output budget (common.config.answer_cap)."""

from __future__ import annotations

from common.config import answer_cap


def test_each_section_gets_its_own_budget():
    assert answer_cap("policy_declarations", "policy") == 4096
    assert answer_cap("policy_schedule", "policy") == 12288
    assert answer_cap("policy_endorsements", "policy") == 3072
    assert answer_cap(None, "policy") == answer_cap("extract", "policy")


def test_a_request_limit_never_exceeds_the_configured_one():
    from inference_core.model_runner import _request_config
    from inference_core.runner_config import RunnerConfig

    config = RunnerConfig(max_new_tokens=8192)
    assert _request_config(config, None, 4096).max_new_tokens == 4096
    assert _request_config(config, None, 20_000).max_new_tokens == 8192
    assert _request_config(config, None, None) is config


def test_validation_and_serving_send_the_section_cap():
    from artifact_registry.blob_client import BlobClient, InMemoryBackend
    from evaluation.validation_generation import generate_validation
    from inference_core import model_runner as M

    seen = []
    backend = M.EchoBackend('{"a":1}')
    real = backend.generate_batch
    backend.generate_batch = lambda msgs, configs, adapter=None: (
        seen.extend(c.max_new_tokens for c in configs), real(msgs, configs, adapter))[1]
    client = BlobClient(backend=InMemoryBackend(), container="main", raw_container="raw")
    model = M.load_model("base", client, backend_impl=backend)
    row = {"source_id": "p1", "doc_type": "policy", "task": "policy_declarations", "messages": [
        {"role": "user", "content": [{"type": "text", "text": "<page 1 of 1>\nx"}]},
        {"role": "assistant", "content": '{"a": 1}'}]}
    generate_validation([row], model, constrain=False)
    assert seen == [4096]

    from serving.pipeline import _window_answer_cap
    assert _window_answer_cap("decl", "policy") == 4096

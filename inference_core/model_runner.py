"""Loading a model version and generating from it — the shared primitive.

Evaluation (SPEC_08), calibration (SPEC_09), serving (SPEC_11) and testing
(SPEC_12) all run the model through here. Building it once, below all four, is
what breaks the dependency cycle they would otherwise form, and what makes
"test == prod" a property of the code rather than a discipline.

Scope is deliberately narrow: **resolve a version, generate, return text and
logprobs.** No confidence calibration, no document-type classification, no
adapter routing — those are higher-level concerns that import this module, never
the other way round.

Backends are swappable (vLLM for serving, HF for local eval, GGUF for the
portable path) behind one interface, so the same calls work everywhere. Logprobs
are mandatory-capable: a backend that cannot return them cannot support the
confidence signal the whole pipeline depends on (arch §5).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from artifact_registry.blob_client import BlobClient
from inference_core.runner_config import Backend, RunnerConfig, load_runner_config
from registry_utils.query_registry import ResolvedModel, resolve_model_version

log = logging.getLogger(__name__)


class ModelRunnerError(RuntimeError):
    """Raised when a model cannot be loaded or generation fails."""


@dataclass
class Generation:
    """One generation, with everything confidence needs."""

    text: str
    tokens: list[str] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    finish_reason: str = "stop"
    generation_fingerprint: str = ""
    latency_ms: float | None = None

    @property
    def has_logprobs(self) -> bool:
        return bool(self.tokens) and len(self.tokens) == len(self.token_logprobs)

    def assert_logprobs(self) -> None:
        """Fail loudly when logprobs are missing.

        Silently returning an extraction with no confidence would hand the caller
        something that looks like a clean result and cannot be risk-routed.
        """
        if not self.has_logprobs:
            raise ModelRunnerError(
                f"generation returned no usable logprobs ({len(self.tokens)} tokens, "
                f"{len(self.token_logprobs)} logprobs). Per-field confidence is derived from "
                "them (arch §5); without them nothing downstream can route low-confidence "
                "fields to review."
            )

    def truncated(self) -> bool:
        """Whether generation hit the token limit.

        Worth surfacing: on a long Loss Run, truncation silently drops claim rows
        — a recall failure that per-field confidence cannot see, because the
        missing rows produced no tokens.
        """
        return self.finish_reason in ("length", "max_tokens")


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------

def assert_tokens_reconstruct(text: str, tokens: list[str], where: str) -> None:
    """Assert the token list rebuilds the generated text exactly.

    **This is the invariant the whole confidence signal rests on.**
    :func:`inference_core.span_map.map_field_spans` locates each field by
    character offset and then attributes the tokens covering that range. If the
    tokens do not concatenate back to the same string — one dropped special
    token, one decode that stripped whitespace — every offset after the
    divergence is wrong, and confidence is attributed to the wrong field.

    Nothing downstream can detect that. The numbers stay in range, the schema
    still validates, and a field the model was unsure about reports the
    confidence of its neighbour. So it is checked here, once, at the only place
    that knows both halves.
    """
    rebuilt = "".join(tokens)
    if rebuilt == text:
        return

    # Locate the first divergence: "they differ" is not actionable on a 4,000
    # character generation.
    index = next(
        (i for i, (a, b) in enumerate(zip(rebuilt, text, strict=False)) if a != b),
        min(len(rebuilt), len(text)),
    )
    raise ModelRunnerError(
        f"{where}: the returned tokens do not reconstruct the generated text "
        f"({len(rebuilt)} chars from {len(tokens)} tokens vs {len(text)} chars of text; "
        f"first difference at {index}: {rebuilt[max(0, index - 30):index + 30]!r} vs "
        f"{text[max(0, index - 30):index + 30]!r}). Per-field confidence is attributed by "
        "character offset, so a mismatch here silently gives each field its neighbour's "
        "confidence — with no error anywhere downstream. Decode without skipping special "
        "tokens, and return the per-token strings the decoder actually produced."
    )


class ModelBackend(ABC):
    """What every backend must provide."""

    @abstractmethod
    def generate(self, messages: list[dict[str, Any]], config: RunnerConfig,
                 adapter: str | None = None) -> Generation: ...

    @abstractmethod
    def supports_logprobs(self) -> bool: ...

    def generate_batch(
        self,
        messages_list: list[list[dict[str, Any]]],
        configs: list[RunnerConfig],
        adapter: str | None = None,
    ) -> list[Generation | Exception]:
        """Several independent requests. One result per request, in order.

        A failure is returned in its slot rather than raised, so one bad request
        does not lose the others. The default runs them one at a time; a backend
        that can batch (vLLM) overrides it and runs them together.
        """
        out: list[Generation | Exception] = []
        for messages, config in zip(messages_list, configs, strict=True):
            try:
                out.append(self.generate(messages, config, adapter=adapter))
            except Exception as exc:  # noqa: BLE001 - reported per request, not raised
                out.append(exc)
        return out


class EchoBackend(ModelBackend):
    """Deterministic stub for tests — no GPU, no weights, no network.

    Returns a canned response with synthetic tokens that reconstruct the text
    exactly, so :mod:`inference_core.span_map` can be exercised end to end.
    """

    def __init__(self, response: str = '{"line_of_business":[]}', chunk: int = 4) -> None:
        self.response = response
        self.chunk = chunk
        self.calls: list[dict[str, Any]] = []

    def supports_logprobs(self) -> bool:
        return True

    def generate(self, messages, config, adapter=None) -> Generation:
        self.calls.append({"messages": messages, "adapter": adapter,
                           "temperature": config.temperature,
                           "json_schema": config.json_schema})
        tokens = [self.response[i:i + self.chunk] for i in range(0, len(self.response), self.chunk)]
        return Generation(
            text=self.response,
            tokens=tokens,
            token_logprobs=[-0.01 * (i % 5) for i in range(len(tokens))],
            generation_fingerprint=config.fingerprint(),
        )


class VLLMBackend(ModelBackend):
    """vLLM — the serving path. Multi-LoRA hot-swap, OpenAI-compatible."""

    def __init__(self, resolved: ResolvedModel, config: RunnerConfig) -> None:
        self.resolved = resolved
        self.config = config
        self._engine = None

    def supports_logprobs(self) -> bool:
        return True

    def _load(self, config: RunnerConfig):
        """Build the engine once and keep it. A cold start per request would
        reload 16GB of weights for every document."""
        if self._engine is not None:
            return self._engine

        from vllm import LLM

        # Checked where the engine is built, which is the only place the mode can
        # still be changed. The comment below used to claim this and nothing did.
        config.assert_logprobs_are_the_models_own()
        model_path = self.resolved.get("merged_model") or self.resolved.get("base_model")
        if not model_path:
            raise ModelRunnerError(
                f"{self.resolved.get('tag')} resolves to no servable weights "
                f"({dict(self.resolved)}). vLLM serves a merged model or the base; a bare adapter "
                "stack has to be merged first (SPEC_10)."
            )

        self._engine = LLM(
            model=model_path,
            enable_lora=config.enable_lora,
            max_loras=config.max_loras,
            max_lora_rank=config.max_lora_rank,
            max_model_len=config.max_seq_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            seed=config.seed,
            # Engine-level, not per request: without it the confidence features
            # read the post-mask distribution under structured decoding (§5.1).
            logprobs_mode=config.logprobs_mode,
        )
        log.info("vLLM engine up on %s (lora=%s)", model_path, config.enable_lora)
        return self._engine

    @staticmethod
    def _imports():
        try:
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError as exc:  # pragma: no cover - optional heavy dep
            raise ModelRunnerError(
                'vLLM is not installed. Install the [serve] extra on the pod: pip install -e ".[serve]"'
            ) from exc
        return SamplingParams, LoRARequest

    @staticmethod
    def _sampling_params(config: RunnerConfig):
        SamplingParams, _ = VLLMBackend._imports()
        sampling: dict[str, Any] = {
            "temperature": config.temperature,
            "top_p": config.top_p,
            "max_tokens": config.max_new_tokens,
            # Requested per token, not per sequence: confidence is per field, and
            # a sequence-level score cannot be attributed to one.
            "logprobs": config.top_logprobs if config.logprobs else None,
            "seed": config.seed,
        }

        # Structured decoding against the target schema (arch v2.1 §13).
        # Guarantees structural validity and stops an enum field taking an
        # invalid value — the model cannot emit a doc type or LoB outside the
        # enum, because those tokens are masked at decode time.
        #
        # THE CAVEAT THAT MATTERS: masking changes the distribution the sampler
        # sees, so `logprobs_mode: raw_logprobs` must be set on the engine or
        # every confidence feature is computed on the post-mask distribution
        # rather than the model's own (§5.1). That is an engine-level setting,
        # asserted at load, not a per-request one — a calibrator fitted on masked
        # logprobs describes a distribution the model never produced.
        if getattr(config, "json_schema", None):
            try:
                from vllm.sampling_params import GuidedDecodingParams

                sampling["guided_decoding"] = GuidedDecodingParams(json=config.json_schema)
            except ImportError:  # pragma: no cover - older vLLM
                log.warning(
                    "this vLLM build exposes no GuidedDecodingParams, so schema validity is "
                    "NOT guaranteed at decode time and the SPEC_07 audit gate is the only "
                    "thing catching an invalid extraction (arch v2.1 §13)."
                )
        return SamplingParams(**{k: v for k, v in sampling.items() if v is not None})

    @staticmethod
    def _lora(adapter: str | None):
        _, LoRARequest = VLLMBackend._imports()
        # Hot-swap rather than reload. The adapter is a per-request argument
        # precisely so one engine serves every document type (arch §4).
        return LoRARequest(adapter, abs(hash(adapter)) % (10 ** 8), adapter) if adapter else None

    @staticmethod
    def _to_generation(output: Any, config: RunnerConfig, latency_ms: float) -> Generation:
        if not output or not output.outputs:
            raise ModelRunnerError("vLLM returned no completion for this request")
        completion = output.outputs[0]

        tokens, logprobs = [], []
        token_ids = list(getattr(completion, "token_ids", None) or [])
        for index, step in enumerate(completion.logprobs or []):
            # Each step maps token_id -> Logprob. The sampled token is looked up
            # by its id, not as the rank-1 entry: under raw_logprobs with
            # structured decoding the sampled token is the best VALID one, which
            # need not be rank 1 in the unmasked distribution — picking rank 1
            # recorded a token that was never emitted and broke reconstruction.
            chosen = step.get(token_ids[index]) if index < len(token_ids) else None
            if chosen is None:
                chosen = next((lp for lp in step.values() if getattr(lp, "rank", 1) == 1), None)
            if chosen is None:
                continue
            tokens.append(chosen.decoded_token)
            logprobs.append(float(chosen.logprob))

        text = completion.text
        if tokens:
            assert_tokens_reconstruct(text, tokens, "vLLM")

        return Generation(
            text=text,
            tokens=tokens,
            token_logprobs=logprobs,
            finish_reason=completion.finish_reason or "stop",
            generation_fingerprint=config.fingerprint(),
            latency_ms=round(latency_ms, 1),
        )

    def generate(self, messages, config, adapter=None) -> Generation:
        import time

        engine = self._load(config)
        params = self._sampling_params(config)
        started = time.perf_counter()
        outputs = engine.chat(messages, params, lora_request=self._lora(adapter))
        latency_ms = (time.perf_counter() - started) * 1000
        return self._to_generation(outputs[0] if outputs else None, config, latency_ms)

    def generate_batch(self, messages_list, configs, adapter=None):
        """All requests in ONE ``engine.chat`` call.

        vLLM schedules them together (continuous batching), so a policy's windows
        run concurrently rather than one after another. Each request keeps its
        own sampling parameters — every window is constrained to a different
        schema slice. The shared wall time is reported on each result, because
        that is how long each one actually waited.
        """
        import time

        if not messages_list:
            return []
        engine = self._load(configs[0])
        params = [self._sampling_params(config) for config in configs]
        started = time.perf_counter()
        try:
            outputs = engine.chat(list(messages_list), params, lora_request=self._lora(adapter))
        except Exception as exc:  # noqa: BLE001 - the whole batch failed; say so per slot
            return [exc for _ in messages_list]
        latency_ms = (time.perf_counter() - started) * 1000

        results: list[Generation | Exception] = []
        for index, config in enumerate(configs):
            try:
                output = outputs[index] if index < len(outputs) else None
                results.append(self._to_generation(output, config, latency_ms))
            except Exception as exc:  # noqa: BLE001 - reported per request
                results.append(exc)
        return results


class HFBackend(ModelBackend):
    """Transformers — local evaluation, and the fallback when vLLM cannot serve."""

    def __init__(self, resolved: ResolvedModel, config: RunnerConfig) -> None:
        self.resolved = resolved
        self.config = config
        self._model = None
        self._processor = None

    def supports_logprobs(self) -> bool:
        return True   # generate(output_scores=True, return_dict_in_generate=True)

    def _load(self, config: RunnerConfig):
        """Load model and processor once, applying the adapter stack if present."""
        if self._model is not None:
            return self._model, self._processor

        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        base = self.resolved.get("merged_model") or self.resolved.get("base_model")
        if not base:
            raise ModelRunnerError(
                f"{self.resolved.get('tag')} resolves to no weights ({dict(self.resolved)})"
            )

        model = AutoModelForVision2Seq.from_pretrained(
            base, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )

        # Foundation first, then the per-type LoRA on top — the order they were
        # trained in. Reversing it produces a different model from the one that
        # was evaluated (arch §4).
        for adapter_path in (self.resolved.get("foundation_adapter"),
                             self.resolved.get("type_adapter")):
            if not adapter_path:
                continue
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter_path)
            log.info("applied adapter %s", adapter_path)

        model.eval()
        self._model = model
        self._processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
        return self._model, self._processor

    def generate(self, messages, config, adapter=None) -> Generation:
        import time

        # Checked before the dependency import: asking this backend to hot-swap
        # an adapter is a contract violation whether or not torch is installed,
        # and reporting it as a missing dependency sends the reader to the wrong
        # problem entirely.
        if adapter and adapter not in (self.resolved.get("type_adapter") or ""):
            # HF applies adapters at load time, not per request. Silently ignoring
            # a mismatch would serve one document type's weights under another's
            # routing decision.
            raise ModelRunnerError(
                f"this backend was loaded with adapter {self.resolved.get('type_adapter')!r} and "
                f"was asked to generate with {adapter!r}. Transformers binds adapters at load "
                "time; per-request hot-swap is the vLLM path (SPEC_11). Load a runner per adapter, "
                "or serve through vLLM."
            )

        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional heavy dep
            raise ModelRunnerError(
                'transformers/torch are not installed. Install the [train] extra: pip install -e ".[train]"'
            ) from exc

        model, processor = self._load(config)
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        started = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                do_sample=not config.is_greedy,
                temperature=config.temperature if not config.is_greedy else None,
                top_p=config.top_p if not config.is_greedy else None,
                output_scores=config.logprobs,
                return_dict_in_generate=True,
            )
        latency_ms = (time.perf_counter() - started) * 1000

        generated_ids = out.sequences[0][prompt_len:]
        # Special tokens are NOT skipped: the token strings have to concatenate
        # back to the decoded text, and skipping them breaks that by exactly the
        # characters they occupy.
        text = processor.decode(generated_ids, skip_special_tokens=True)

        tokens: list[str] = []
        logprobs: list[float] = []
        if config.logprobs and getattr(out, "scores", None):
            for step, token_id in zip(out.scores, generated_ids, strict=False):
                token = processor.decode([token_id], skip_special_tokens=True)
                if not token:
                    continue      # a special token contributes no text
                tokens.append(token)
                logprobs.append(
                    float(torch.log_softmax(step[0].float(), dim=-1)[token_id])
                )

        if tokens:
            assert_tokens_reconstruct(text, tokens, "Transformers")

        finished = generated_ids.shape[-1] >= config.max_new_tokens
        return Generation(
            text=text,
            tokens=tokens,
            token_logprobs=logprobs,
            finish_reason="length" if finished else "stop",
            generation_fingerprint=config.fingerprint(),
            latency_ms=round(latency_ms, 1),
        )


class GGUFBackend(ModelBackend):
    """llama.cpp — the portable/edge path.

    Qwen3-VL is multimodal, so a GGUF export needs the ``mmproj`` projector file
    alongside the quantized LLM. Whether llama.cpp supports it for this model is
    unverified (arch §13a), which is why GGUF is not the primary serving route.
    """

    def __init__(self, resolved: ResolvedModel, config: RunnerConfig) -> None:
        self.resolved = resolved
        self.config = config

    def supports_logprobs(self) -> bool:
        return True

    def generate(self, messages, config, adapter=None) -> Generation:
        raise NotImplementedError(
            "GGUF serving is the portable path, not the first-cycle one. Confirm llama.cpp "
            "supports Qwen3-VL with an mmproj file before wiring it (arch §13a)."
        )


#: Constructors, not classes: every real backend takes ``(resolved, config)``
#: while the ABC itself takes nothing, so the callable type is the accurate one.
_BACKENDS: dict[Backend, Callable[[ResolvedModel, RunnerConfig], ModelBackend]] = {
    "vllm": VLLMBackend,
    "hf": HFBackend,
    "gguf": GGUFBackend,
}


# --------------------------------------------------------------------------
# Loading and generating
# --------------------------------------------------------------------------

@dataclass
class LoadedModel:
    """A resolved model version plus the backend that runs it."""

    tag: str
    resolved: ResolvedModel
    backend: ModelBackend
    config: RunnerConfig

    @property
    def is_base(self) -> bool:
        """Whether this is the untuned base with no adapter.

        The shared path for the pilot's zero-shot baseline (SPEC_15), day-zero
        pre-annotation (SPEC_04), and ``extract --model base`` (SPEC_13).
        """
        return self.resolved.get("kind") == "base"


def load_model(
    tag: str,
    client: BlobClient,
    *,
    doc_type: str | None = None,
    backend: Backend | None = None,
    quant_format: str | None = None,
    config: RunnerConfig | None = None,
    backend_impl: ModelBackend | None = None,
) -> LoadedModel:
    """Resolve a version tag and prepare it for generation.

    ``tag`` accepts ``base`` as a first-class value — the untuned model with no
    adapter — as well as ``v1``, ``v2``, and so on.

    Args:
        backend_impl: inject a backend directly, for tests. Production callers
            leave this unset and get the configured backend.
    """
    config = config or load_runner_config(backend)
    resolved = resolve_model_version(tag, client, doc_type=doc_type, quant_format=quant_format)

    if backend_impl is not None:
        impl = backend_impl
    else:
        backend_cls = _BACKENDS.get(config.backend)
        if backend_cls is None:
            raise ModelRunnerError(
                f"unknown backend {config.backend!r}; expected one of {sorted(_BACKENDS)}"
            )
        impl = backend_cls(resolved, config)

    if config.logprobs and not impl.supports_logprobs():
        raise ModelRunnerError(
            f"backend {config.backend!r} cannot return logprobs, but per-field confidence "
            "depends on them (arch §5). Use a backend that can, or accept that this run "
            "produces no confidence signal — and say so explicitly rather than by omission."
        )

    log.info(
        "loaded %s (kind=%s, backend=%s, staged=%s)",
        tag, resolved.get("kind"), config.backend, resolved.get("from_staging"),
    )
    return LoadedModel(tag=tag, resolved=resolved, backend=impl, config=config)


def generate(
    model: LoadedModel,
    messages: list[dict[str, Any]],
    *,
    adapter: str | None = None,
    want_logprobs: bool = True,
    json_schema: dict[str, Any] | None = None,
) -> Generation:
    """Generate from prepared messages.

    Args:
        adapter: per-type LoRA to apply for this request. ``None`` means
            Foundation-only — the classifier's low-confidence fallback path
            (arch §4a), not an error.
        json_schema: the resolved target schema to constrain this request to.
            Per request because it is per document type.
    """
    import dataclasses
    import time

    config = (
        dataclasses.replace(model.config, json_schema=json_schema) if json_schema else model.config
    )
    started = time.perf_counter()
    try:
        result = model.backend.generate(messages, config, adapter=adapter)
    except NotImplementedError:
        raise
    except Exception as exc:
        raise ModelRunnerError(f"generation failed for {model.tag}: {exc}") from exc

    result.latency_ms = round((time.perf_counter() - started) * 1000, 1)
    if not result.generation_fingerprint:
        result.generation_fingerprint = model.config.fingerprint()

    if want_logprobs:
        result.assert_logprobs()
    if result.truncated():
        log.warning(
            "generation for %s hit the token limit — output may be incomplete. On a long "
            "Loss Run this silently drops claim rows, which per-field confidence cannot detect.",
            model.tag,
        )
    return result


def generate_batch(
    model: LoadedModel,
    requests: list[tuple[list[dict[str, Any]], dict[str, Any] | None]],
    *,
    adapter: str | None = None,
    want_logprobs: bool = True,
) -> list[Generation | ModelRunnerError]:
    """Several independent generations, run together where the backend can.

    ``requests`` is ``[(messages, json_schema), ...]`` — each constrained to its
    own schema, as a policy's windows are. One result per request, in order; a
    failed request is a :class:`ModelRunnerError` in its slot, never raised, so
    one window cannot lose the others. The checks :func:`generate` applies —
    logprobs present — apply to each result.
    """
    import dataclasses

    configs = [
        dataclasses.replace(model.config, json_schema=schema) if schema else model.config
        for _messages, schema in requests
    ]
    raw = model.backend.generate_batch(
        [messages for messages, _schema in requests], configs, adapter=adapter
    )

    out: list[Generation | ModelRunnerError] = []
    for result in raw:
        if isinstance(result, Exception):
            out.append(
                result if isinstance(result, ModelRunnerError)
                else ModelRunnerError(f"generation failed for {model.tag}: {result}")
            )
            continue
        if not result.generation_fingerprint:
            result.generation_fingerprint = model.config.fingerprint()
        if want_logprobs and not result.has_logprobs:
            try:
                result.assert_logprobs()
            except ModelRunnerError as exc:
                out.append(exc)
                continue
        out.append(result)
    return out

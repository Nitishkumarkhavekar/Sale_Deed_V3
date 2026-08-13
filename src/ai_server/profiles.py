"""Automatic inference-profile selection for the locally trained deeds model.

Reads the geometry of *your* checkpoint from its `config.json` - nothing is
hardcoded to a published model - measures the machine, then picks the
highest-fidelity configuration that provably fits in VRAM.

Design principle: fidelity is maximised subject to a hard context floor. A
configuration that cannot hold a real sale deed is worthless no matter how
precise its weights, so context is a constraint and quantisation is the
variable we trade.

Corpus facts driving the context floor (measured over `test/OCR saledeeds`):
    minimum   5.4k prompt tokens
    median    9.4k
    maximum   33k
Output is short: ~664 tokens average, ~1300 observed maximum.

Scaling: every number below is derived from measured free VRAM. Fit a larger
card and the selector climbs to a higher-fidelity rung on its own - at 16 GB it
selects BF16, where quantisation loss is exactly zero. No code changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .hardware import GIB, MIB, HardwareInfo, detect

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: VRAM fraction held back so the driver never hits an allocation failure.
#: The deployment requirement specifies 8%.
VRAM_HEADROOM_FRACTION = 0.08

#: Preferred context. Covers the corpus median with generous room for output.
TARGET_CTX = 24_576

#: Absolute floor. Below this the median deed no longer fits and extraction
#: would silently truncate, so we refuse rather than degrade quietly.
MIN_CTX = 16_384

#: Reserve for generated output, subtracted when reporting usable prompt room.
OUTPUT_RESERVE_TOKENS = 2_048

#: Ceiling on how much of the (already headroom-reduced) budget a profile may
#: claim. A configuration computed at 99% full will OOM the moment the driver,
#: the compositor or a background process allocates anything at all. Estimates
#: here are approximations, so leaving slack on top of the headroom is what
#: actually delivers the "never OOM" requirement.
MAX_BUDGET_UTILISATION = 0.95

#: Windows WDDM CUDA context plus driver allocations. Empirical.
CUDA_CONTEXT_BYTES = 250 * MIB

#: llama.cpp compute buffers (attention workspace, logits). Empirical for a 4B
#: at the default batch size, with headroom folded in.
COMPUTE_BUFFER_BYTES = 350 * MIB

#: Extra compute buffer per concurrent slot beyond the first.
PER_SLOT_BYTES = 64 * MIB


@dataclass(frozen=True)
class Quant:
    """One rung on the fidelity ladder."""

    name: str
    bits_per_weight: float
    #: True only when weights are bit-identical to the trained checkpoint.
    lossless: bool
    note: str


#: Highest fidelity first. The selector returns the first rung that fits.
QUANT_LADDER: tuple[Quant, ...] = (
    Quant("BF16", 16.0, True, "identical to your trained weights - zero loss"),
    Quant("Q8_0", 8.5, False, "loss below measurement noise"),
    Quant("Q6_K", 6.6, False, "negligible loss"),
    Quant("Q5_K_M", 5.8, False, "small loss; verify on the deed corpus"),
    Quant("Q4_K_M", 5.1, False, "measurable loss; verify on the deed corpus"),
)

#: KV cache element sizes in bytes. f16 first - quantised KV hurts exact-copy
#: accuracy (Aadhaar digits, PAN strings) more than quantised weights do.
KV_TYPES: tuple[tuple[str, float], ...] = (("f16", 2.0), ("q8_0", 1.09))


# ---------------------------------------------------------------------------
# Model geometry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelGeometry:
    """Shape of the checkpoint, read from its own config.json."""

    name: str
    n_layers: int
    n_full_attention_layers: int
    n_sliding_layers: int
    sliding_window: int
    n_kv_heads: int
    n_heads: int
    head_dim: int
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    has_vision_tower: bool

    @property
    def embedding_params(self) -> int:
        return self.vocab_size * self.hidden_size

    @property
    def params_per_layer(self) -> int:
        attn = (
            self.hidden_size * self.n_heads * self.head_dim  # q
            + self.hidden_size * self.n_kv_heads * self.head_dim  # k
            + self.hidden_size * self.n_kv_heads * self.head_dim  # v
            + self.n_heads * self.head_dim * self.hidden_size  # o
        )
        mlp = 3 * self.hidden_size * self.intermediate_size  # gate, up, down
        return attn + mlp

    @property
    def total_params(self) -> int:
        """Text-tower parameters. The vision tower is not served."""
        return self.embedding_params + self.n_layers * self.params_per_layer

    def weight_bytes(self, quant: Quant) -> int:
        return int(self.total_params * quant.bits_per_weight / 8)

    def kv_bytes(self, n_ctx: int, kv_elem_bytes: float) -> int:
        """KV cache size, honouring Gemma 3 interleaved sliding-window attention.

        Only the full-attention layers scale with context; the sliding layers
        are capped at their window. For this model that is 5 of 34 layers, which
        is the entire reason a 24k context is affordable on a 4 GB card.
        """
        per_token_per_layer = 2 * self.n_kv_heads * self.head_dim * kv_elem_bytes
        scaling = self.n_full_attention_layers * n_ctx
        capped = self.n_sliding_layers * min(n_ctx, self.sliding_window)
        return int((scaling + capped) * per_token_per_layer)


def load_geometry(model_dir: str | Path) -> ModelGeometry:
    """Parse geometry out of the checkpoint's own config.json."""
    path = Path(model_dir)
    config_path = path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"no config.json in {path}")

    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    has_vision = "vision_config" in cfg
    text = cfg.get("text_config", cfg)  # multimodal wrapper nests the text config

    layer_types = text.get("layer_types") or []
    n_layers = int(text["num_hidden_layers"])
    if layer_types:
        n_full = sum(1 for t in layer_types if t == "full_attention")
    else:
        n_full = n_layers  # no interleaving declared: assume all full attention
    window = int(text.get("sliding_window") or 0) or 10**9

    n_heads = int(text["num_attention_heads"])
    head_dim = int(text.get("head_dim") or text["hidden_size"] // n_heads)

    return ModelGeometry(
        name=path.name,
        n_layers=n_layers,
        n_full_attention_layers=n_full,
        n_sliding_layers=n_layers - n_full,
        sliding_window=window,
        n_kv_heads=int(text["num_key_value_heads"]),
        n_heads=n_heads,
        head_dim=head_dim,
        hidden_size=int(text["hidden_size"]),
        intermediate_size=int(text["intermediate_size"]),
        vocab_size=int(text["vocab_size"]),
        has_vision_tower=has_vision,
    )


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    """A concrete, validated runtime configuration."""

    device: str  # "cuda" | "cpu"
    quant: Quant
    n_ctx: int
    kv_type: str
    n_gpu_layers: int
    n_parallel: int
    n_threads: int
    gpu_uuid: str | None

    weight_bytes: int
    kv_bytes: int
    overhead_bytes: int
    budget_bytes: int

    reason: str
    warnings: tuple[str, ...] = ()

    @property
    def total_bytes(self) -> int:
        return self.weight_bytes + self.kv_bytes + self.overhead_bytes

    @property
    def prompt_capacity(self) -> int:
        """Prompt tokens available per request after reserving output room."""
        return self.n_ctx // self.n_parallel - OUTPUT_RESERVE_TOKENS

    def explain(self) -> str:
        pct = 100 * self.total_bytes / self.budget_bytes if self.budget_bytes else 0
        lines = [
            "Selected inference profile",
            f"  Device        : {self.device.upper()}"
            + (f"  (pinned to {self.gpu_uuid})" if self.gpu_uuid else ""),
            f"  Quantisation  : {self.quant.name} - {self.quant.note}",
            f"  Context       : {self.n_ctx:,} tokens "
            f"({self.prompt_capacity:,} usable for prompt)",
            f"  KV cache      : {self.kv_type}",
            f"  GPU layers    : {self.n_gpu_layers}",
            f"  Slots         : {self.n_parallel}",
            f"  Threads       : {self.n_threads}",
            "",
            "  VRAM budget",
            f"    weights     : {self.weight_bytes / GIB:7.2f} GiB",
            f"    kv cache    : {self.kv_bytes / GIB:7.2f} GiB",
            f"    overhead    : {self.overhead_bytes / GIB:7.2f} GiB",
            f"    {'-' * 26}",
            f"    total       : {self.total_bytes / GIB:7.2f} GiB "
            f"of {self.budget_bytes / GIB:.2f} GiB usable ({pct:.0f}%)",
            "",
            f"  Rationale     : {self.reason}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING       : {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def _overhead(n_parallel: int) -> int:
    return CUDA_CONTEXT_BYTES + COMPUTE_BUFFER_BYTES + (n_parallel - 1) * PER_SLOT_BYTES


def _fits(geo: ModelGeometry, quant: Quant, n_ctx: int, kv_elem: float,
          n_parallel: int, budget: int) -> tuple[bool, int, int, int]:
    """Does this configuration fit, with slack left over? Returns the breakdown."""
    weights = geo.weight_bytes(quant)
    kv = geo.kv_bytes(n_ctx, kv_elem)
    overhead = _overhead(n_parallel)
    ceiling = budget * MAX_BUDGET_UTILISATION
    return (weights + kv + overhead) <= ceiling, weights, kv, overhead


def select_profile(
    model_dir: str | Path,
    hw: HardwareInfo | None = None,
    *,
    allow_cpu: bool = True,
) -> Profile:
    """Choose the highest-fidelity configuration that fits the detected hardware."""
    hw = hw or detect()
    geo = load_geometry(model_dir)
    threads = max(1, hw.physical_cores)
    warnings: list[str] = []

    gpu = hw.primary_gpu
    if gpu is None:
        if not allow_cpu:
            raise RuntimeError("no CUDA GPU detected and CPU fallback is disabled")
        return _cpu_profile(geo, threads, ["no CUDA GPU detected - CPU fallback"])

    budget = int(gpu.free_bytes * (1.0 - VRAM_HEADROOM_FRACTION))
    if gpu.is_display_gpu:
        warnings.append(
            f"{gpu.name} is driving a display ({gpu.used_bytes / MIB:.0f} MiB in use); "
            "that memory is unavailable to inference"
        )

    # Search order encodes the accuracy priorities, outermost first:
    #   1. context      - a config that truncates real deeds is useless
    #   2. KV precision - quantised KV degrades exact-copy fields (Aadhaar, PAN)
    #                     more than quantised weights do
    #   3. weight precision
    for n_ctx in (TARGET_CTX, 20_480, MIN_CTX):
        for kv_name, kv_elem in KV_TYPES:
            for quant in QUANT_LADDER:
                ok, weights, kv, overhead = _fits(geo, quant, n_ctx, kv_elem, 1, budget)
                if not ok:
                    continue

                n_parallel = _slots(geo, quant, n_ctx, kv_elem, budget)
                if n_parallel > 1:
                    kv = geo.kv_bytes(n_ctx * n_parallel, kv_elem)
                    overhead = _overhead(n_parallel)

                reason = (
                    f"highest-fidelity rung fitting {budget / GIB:.2f} GiB "
                    f"({100 * VRAM_HEADROOM_FRACTION:.0f}% headroom held back) "
                    f"at {n_ctx:,}-token context"
                )
                if n_ctx < TARGET_CTX:
                    warnings.append(
                        f"context reduced to {n_ctx:,} to fit; deeds longer than "
                        f"~{n_ctx - OUTPUT_RESERVE_TOKENS:,} prompt tokens will be rejected"
                    )
                if not quant.lossless:
                    warnings.append(
                        f"{quant.name} is lossy relative to your trained weights - "
                        "measure against the BF16 baseline before production use"
                    )
                if kv_name != "f16":
                    warnings.append(
                        "quantised KV cache: verify exact-copy fields "
                        "(Aadhaar, PAN) against the baseline"
                    )

                return Profile(
                    device="cuda",
                    quant=quant,
                    n_ctx=n_ctx * n_parallel,
                    kv_type=kv_name,
                    n_gpu_layers=geo.n_layers + 1,  # +1 for the output layer
                    n_parallel=n_parallel,
                    n_threads=threads,
                    gpu_uuid=gpu.uuid,
                    weight_bytes=weights,
                    kv_bytes=kv,
                    overhead_bytes=overhead,
                    budget_bytes=budget,
                    reason=reason,
                    warnings=tuple(warnings),
                )

    # Nothing fits fully on the GPU. Partial offload keeps most layers on the
    # card and the remainder on the CPU - slower, but it still uses the GPU.
    partial = _partial_offload(geo, gpu, budget, threads, warnings)
    if partial is not None:
        return partial

    if not allow_cpu:
        raise RuntimeError(
            f"model does not fit in {budget / GIB:.2f} GiB and CPU fallback is disabled"
        )
    warnings.append("model does not fit in VRAM at the minimum context - CPU fallback")
    return _cpu_profile(geo, threads, warnings)


def _slots(geo: ModelGeometry, quant: Quant, n_ctx: int, kv_elem: float, budget: int) -> int:
    """How many concurrent sequences fit.

    llama.cpp divides total context across slots, so N slots each needing
    `n_ctx` tokens requires `N * n_ctx` of context and proportionally more KV.

    Capped at 4: this workload is prefill-bound with 5k-33k token prompts, and a
    single long prefill already saturates the GPU. Extra slots buy little and
    cost KV that is better spent on context.
    """
    for n in (4, 3, 2):
        ok, *_ = _fits(geo, quant, n_ctx * n, kv_elem, n, budget)
        if ok:
            return n
    return 1


def _partial_offload(
    geo: ModelGeometry, gpu, budget: int, threads: int, warnings: list[str]
) -> Profile | None:
    """Put as many layers on the GPU as fit, at the smallest viable context."""
    quant = QUANT_LADDER[-1]  # most compact rung
    kv_name, kv_elem = KV_TYPES[-1]
    n_ctx = MIN_CTX

    kv = geo.kv_bytes(n_ctx, kv_elem)
    overhead = _overhead(1)
    available = budget - kv - overhead
    if available <= 0:
        return None

    bytes_per_layer = int(geo.params_per_layer * quant.bits_per_weight / 8)
    n_gpu_layers = min(geo.n_layers, max(0, available // bytes_per_layer))
    if n_gpu_layers == 0:
        return None

    warnings.append(
        f"only {n_gpu_layers} of {geo.n_layers} layers fit in VRAM; the remainder run "
        "on CPU, which is substantially slower"
    )
    return Profile(
        device="cuda",
        quant=quant,
        n_ctx=n_ctx,
        kv_type=kv_name,
        n_gpu_layers=n_gpu_layers,
        n_parallel=1,
        n_threads=threads,
        gpu_uuid=gpu.uuid,
        weight_bytes=n_gpu_layers * bytes_per_layer,
        kv_bytes=kv,
        overhead_bytes=overhead,
        budget_bytes=budget,
        reason=f"partial offload: {n_gpu_layers}/{geo.n_layers} layers on GPU",
        warnings=tuple(warnings),
    )


def _cpu_profile(geo: ModelGeometry, threads: int, warnings: list[str]) -> Profile:
    quant = QUANT_LADDER[-1]
    kv_name, kv_elem = KV_TYPES[-1]
    warnings = [*warnings, "CPU inference is roughly an order of magnitude slower"]
    return Profile(
        device="cpu",
        quant=quant,
        n_ctx=MIN_CTX,
        kv_type=kv_name,
        n_gpu_layers=0,
        n_parallel=1,
        n_threads=threads,
        gpu_uuid=None,
        weight_bytes=geo.weight_bytes(quant),
        kv_bytes=geo.kv_bytes(MIN_CTX, kv_elem),
        overhead_bytes=COMPUTE_BUFFER_BYTES,
        budget_bytes=0,
        reason="no usable GPU configuration",
        warnings=tuple(warnings),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def ladder_report(model_dir: str | Path, hw: HardwareInfo | None = None) -> str:
    """Show every rung and whether it fits - the 'why not better?' answer."""
    hw = hw or detect()
    geo = load_geometry(model_dir)
    gpu = hw.primary_gpu
    budget = int(gpu.free_bytes * (1.0 - VRAM_HEADROOM_FRACTION)) if gpu else 0

    ceiling = budget * MAX_BUDGET_UTILISATION
    rows = [
        f"Fidelity ladder at {TARGET_CTX:,} context "
        f"(budget {budget / GIB:.2f} GiB, usable ceiling {ceiling / GIB:.2f} GiB "
        f"at {MAX_BUDGET_UTILISATION:.0%})",
        f"  {'quant':<8} {'kv':<5} {'weights':>9} {'kv$':>8} {'total':>9}   verdict",
    ]
    for kv_name, kv_elem in KV_TYPES:
        for quant in QUANT_LADDER:
            weights = geo.weight_bytes(quant)
            kv = geo.kv_bytes(TARGET_CTX, kv_elem)
            total = weights + kv + _overhead(1)
            if total <= ceiling:
                verdict = f"fits ({total / budget:.0%} of budget)"
            elif total <= budget:
                verdict = f"too tight ({total / budget:.0%}) - OOM risk"
            else:
                verdict = f"needs {(total - budget) / GIB:.2f} GiB more"
            rows.append(
                f"  {quant.name:<8} {kv_name:<5} {weights / GIB:8.2f}G {kv / GIB:7.2f}G "
                f"{total / GIB:8.2f}G   {verdict}"
            )
    return "\n".join(rows)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    model = sys.argv[1] if len(sys.argv) > 1 else "AI server/gemma4b"

    hw = detect()
    geo = load_geometry(model)

    print(hw.summary())
    print()
    print(f"Model: {geo.name}")
    print(f"  layers        : {geo.n_layers} "
          f"({geo.n_full_attention_layers} full attention, "
          f"{geo.n_sliding_layers} sliding @ {geo.sliding_window})")
    print(f"  text params   : {geo.total_params / 1e9:.2f} B")
    print(f"  vocab         : {geo.vocab_size:,}")
    if geo.has_vision_tower:
        print("  vision tower  : present but not served (text-only extraction)")
    print()
    print(ladder_report(model, hw))
    print()
    print(select_profile(model, hw).explain())

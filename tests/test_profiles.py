"""The VRAM ladder: what fits on this card, and what it costs.

The most consequential module at low coverage. It decides the quantisation, the
context window and the number of layers offloaded - and it decides them from
*measured free VRAM*, before anything is loaded. Get it wrong in the generous
direction and llama-server starts, serves a few deeds and then dies mid-batch
with a CUDA OOM; get it wrong in the mean direction and a 24 GB card runs the
same tiny context as a laptop.

Tested against synthesised hardware rather than the machine running the tests,
so the ladder's behaviour on cards this project will meet - 4, 8, 16, 24 GB -
is pinned regardless of what is in the box today.
"""

from __future__ import annotations

import pytest

from ai_server.profiles import (
    KV_TYPES,
    MAX_BUDGET_UTILISATION,
    MIN_CTX,
    OUTPUT_RESERVE_TOKENS,
    QUANT_LADDER,
    TARGET_CTX,
    VRAM_HEADROOM_FRACTION,
    ModelGeometry,
    Profile,
    select_profile,
)

GIB = 1024 ** 3


#: Gemma-3-4B's shape, written as a real config.json rather than a constructed
#: ModelGeometry. `select_profile` reads the directory itself, so feeding it one
#: exercises `load_geometry` as well - including the interleaved
#: full/sliding attention layout, which is what makes this model's KV cache
#: much smaller than its layer count suggests.
CONFIG = {
    "num_hidden_layers": 34,
    "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 5 + [
        "sliding_attention"] * 4,
    "sliding_window": 1024,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "hidden_size": 2560,
    "intermediate_size": 10240,
    "vocab_size": 262208,
}


@pytest.fixture()
def model_dir(tmp_path):
    import json

    (tmp_path / "config.json").write_text(json.dumps(CONFIG), encoding="utf-8")
    return tmp_path


def _hardware(vram_gib, *, cores=8, cuda=True):
    """A card of a given size, built from the real dataclasses.

    Not a stub: a hand-rolled object silently lacks whatever the ladder reads
    next - `is_display_gpu` in this case - and turns a genuine result into an
    AttributeError inside the test. The real types fail loudly instead when
    the shape changes.
    """
    from ai_server.hardware import GpuDevice, HardwareInfo

    gpus = () if not cuda else (GpuDevice(
        index=0, uuid="GPU-test", name=f"Synthetic {vram_gib} GiB",
        total_bytes=int(vram_gib * GIB), free_bytes=int(vram_gib * GIB),
        used_bytes=0, compute_capability="8.6"),)
    return HardwareInfo(
        os_name="Windows", cpu_name="Synthetic", logical_cores=cores * 2,
        physical_cores=cores, ram_total_bytes=16 * GIB,
        ram_available_bytes=8 * GIB, cuda_available=cuda,
        driver_version="500.00", cuda_version="12.6", gpus=gpus,
        disks=(), other_adapters=(), warnings=())


class TestTheLadderIsOrdered:
    def test_quantisation_runs_from_lossless_to_lossiest(self, model_dir):
        """The ladder is walked top-down and the first fit wins, so the order
        *is* the preference. Reversed, every machine would get Q4."""
        assert QUANT_LADDER[0].name == "BF16"
        assert QUANT_LADDER[-1].name.startswith("Q4")
        sizes = [q.bits_per_weight for q in QUANT_LADDER]
        assert sizes == sorted(sizes, reverse=True)

    def test_only_bf16_claims_zero_loss(self):
        """Every other rung must admit it is lossy, because that admission is
        what tells an operator to verify against the baseline."""
        lossless = [q for q in QUANT_LADDER if q.lossless]
        assert [q.name for q in lossless] == ["BF16"]

    def test_the_kv_types_are_ordered_by_fidelity(self):
        assert KV_TYPES[0][0] == "f16"
        assert KV_TYPES[0][1] > KV_TYPES[-1][1]


class TestHeadroomIsHeldBack:
    def test_the_budget_never_claims_the_whole_card(self):
        """A card is never entirely free - the display alone holds some - and
        claiming all of it is how a profile that 'fits' OOMs on load."""
        assert 0 < VRAM_HEADROOM_FRACTION < 1
        assert 0 < MAX_BUDGET_UTILISATION < 1

    def test_a_profile_stays_inside_its_own_budget(self, model_dir):
        profile = select_profile(model_dir, _hardware(24))
        used = profile.weight_bytes + profile.kv_bytes + profile.overhead_bytes
        assert used <= profile.budget_bytes, "the profile overcommits the card"

    def test_output_tokens_are_reserved(self, model_dir):
        """The context has to hold the answer as well as the deed."""
        assert OUTPUT_RESERVE_TOKENS > 0


class TestBiggerCardsGetMore:
    @pytest.mark.parametrize("vram", [4, 6, 8, 12, 16, 24, 48])
    def test_every_card_produces_a_usable_profile(self, model_dir, vram):
        profile = select_profile(model_dir, _hardware(vram))
        assert profile.n_ctx >= MIN_CTX or profile.device == "cpu"
        assert profile.n_threads >= 1
        assert profile.reason

    def test_capacity_never_decreases_as_vram_grows(self, model_dir):
        """Monotonic by construction. A 24 GB card returning less than a 16 GB
        one would be a ladder bug, and it would be invisible in production -
        the smaller machine would simply look oddly good."""
        seen = []
        for vram in (4, 8, 16, 24, 48):
            p = select_profile(model_dir, _hardware(vram))
            seen.append((vram, p.weight_bytes + p.kv_bytes))
        values = [v for _, v in seen]
        assert values == sorted(values), f"capacity went backwards: {seen}"

    def test_a_large_card_reaches_the_target_context(self, model_dir):
        profile = select_profile(model_dir, _hardware(48))
        assert profile.n_ctx >= TARGET_CTX

    def test_a_large_card_is_not_forced_onto_the_lossiest_quant(self, model_dir):
        """48 GB has room for the trained weights; serving Q4 there would give
        away accuracy for nothing."""
        profile = select_profile(model_dir, _hardware(48))
        assert profile.quant.name != QUANT_LADDER[-1].name


class TestSmallCards:
    def test_a_four_gib_card_still_gets_a_profile(self, model_dir):
        """The machine this project runs on. It must not raise - it must
        return the best thing that fits, and say what that cost."""
        profile = select_profile(model_dir, _hardware(4))
        assert isinstance(profile, Profile)
        assert profile.reason

    def test_a_tiny_card_falls_back_rather_than_overcommitting(self, model_dir):
        profile = select_profile(model_dir, _hardware(1))
        used = profile.weight_bytes + profile.kv_bytes + profile.overhead_bytes
        assert profile.device == "cpu" or used <= profile.budget_bytes

    def test_no_gpu_means_cpu(self, model_dir):
        profile = select_profile(model_dir, _hardware(0, cuda=False))
        assert profile.device == "cpu"
        assert profile.n_gpu_layers == 0

    def test_refusing_cpu_is_honoured(self, model_dir):
        """A caller that cannot accept a CPU profile must be told, not handed
        one that would take hours per deed."""
        with pytest.raises(Exception):
            select_profile(model_dir, _hardware(0, cuda=False), allow_cpu=False)


class TestTheProfileExplainsItself:
    def test_it_says_why_it_chose_what_it_chose(self, model_dir):
        profile = select_profile(model_dir, _hardware(8))
        assert len(profile.reason) > 10

    def test_a_lossy_quant_warns(self, model_dir):
        """The warning is what tells an operator the exported identifiers
        should be checked against the BF16 baseline."""
        profile = select_profile(model_dir, _hardware(4))
        if not profile.quant.lossless:
            assert any("lossy" in w.lower() or "verify" in w.lower()
                       for w in profile.warnings), profile.warnings

    def test_a_reduced_context_warns(self, model_dir):
        profile = select_profile(model_dir, _hardware(4))
        if profile.n_ctx < TARGET_CTX and profile.device == "cuda":
            assert profile.warnings

    def test_the_gpu_is_pinned_by_uuid(self, model_dir):
        """By UUID, not index: indices are reassigned when a display is
        plugged in, and the integrated adapter must never be selected."""
        profile = select_profile(model_dir, _hardware(16))
        if profile.device == "cuda":
            assert profile.gpu_uuid

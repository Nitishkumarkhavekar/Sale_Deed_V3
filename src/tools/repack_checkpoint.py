"""Losslessly repack the trained deeds checkpoint into a standard text-only model.

WHAT THIS DOES NOT DO
    It does not retrain, fine-tune, quantise, re-initialise or otherwise alter a
    single weight. Every BF16 value is copied byte-for-byte. Only dictionary
    keys are renamed and the unused vision tensors are omitted.

WHY IT IS NEEDED
    The checkpoint stores layers as

        model.language_model.model.layers.N....

    where current Gemma 3 expects

        model.language_model.layers.N....

    That extra `.model.` is a snapshot of a transformers refactor caught midway
    (config.json records transformers_version 5.5.0). Consequences observed:

      * transformers 5.8.x loads it WITHOUT ERROR and silently random-initialises
        the language model - it produces fluent nonsense with no exception.
      * vLLM needs a hand-patched WeightsMapper.
      * llama.cpp's GGUF converter will fail the same way.

    Renaming once at the source fixes every consumer permanently.

    The vision tower is dropped because extraction runs on OCR text: images were
    measured as out-of-distribution and less accurate. That is 437 of 883
    tensors and ~0.84 GB carried through every load and conversion.

MEMORY
    Pure byte-range copy through a bounded buffer: no torch, no safetensors
    package, no numpy, and peak RAM stays at the chunk size. That matters on an
    8 GB machine where the source file alone is 8.6 GB.

USAGE
    python tools/repack_checkpoint.py --dry-run
    python tools/repack_checkpoint.py
    python tools/repack_checkpoint.py --src "AI server/gemma4b" --dst "AI server/gemma4b-text"
"""

from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
import time
from pathlib import Path

# Prefix rewrite: the whole fix, in one rule.
OLD_PREFIX = "model.language_model.model."
NEW_PREFIX = "model."

# Tensor groups that exist only to serve images.
VISION_PREFIXES = ("model.vision_tower.", "model.multi_modal_projector.")

# Copied through unchanged. preprocessor_config.json is deliberately excluded:
# it configures the image processor, which no longer has a tower to feed.
SIDECAR_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")

COPY_CHUNK = 16 * 1024 * 1024
GIB = 1024**3

DTYPE_SIZES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def read_header(path: Path) -> tuple[dict, dict, int]:
    """Return (tensors, metadata, data_start) from a safetensors file."""
    with path.open("rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
    metadata = header.pop("__metadata__", {})
    return header, metadata, 8 + header_len


def classify(tensors: dict) -> tuple[dict, list[str], list[str]]:
    """Split source keys into kept-and-renamed, dropped-vision, and unexpected."""
    keep: dict[str, dict] = {}
    vision: list[str] = []
    unknown: list[str] = []

    for key, info in tensors.items():
        if key.startswith(VISION_PREFIXES):
            vision.append(key)
        elif key.startswith(OLD_PREFIX):
            keep[NEW_PREFIX + key[len(OLD_PREFIX):]] = info
        else:
            unknown.append(key)

    return keep, vision, unknown


def build_config(src_cfg: dict) -> dict:
    """Hoist the nested text config into a standalone Gemma3ForCausalLM config."""
    text = dict(src_cfg.get("text_config", src_cfg))
    text["architectures"] = ["Gemma3ForCausalLM"]
    text["model_type"] = "gemma3_text"
    text.setdefault("tie_word_embeddings", src_cfg.get("tie_word_embeddings", True))
    for inherited in ("dtype", "torch_dtype", "transformers_version"):
        if inherited in src_cfg and inherited not in text:
            text[inherited] = src_cfg[inherited]
    return text


def fix_generation_config(cfg: dict) -> tuple[dict, list[str]]:
    """Make the shipped sampling defaults match how extraction must actually run.

    The checkpoint ships do_sample=true with top_k/top_p. Structured extraction
    requires greedy decoding: any HF call that forgets do_sample=False would
    silently sample. vLLM and llama.cpp take sampling from the request, so this
    only affects the transformers path - but that is exactly the path where the
    mistake is invisible.
    """
    notes: list[str] = []
    out = dict(cfg)
    if out.get("do_sample"):
        out["do_sample"] = False
        notes.append("do_sample true -> false (extraction is greedy)")
    for key in ("top_k", "top_p", "temperature"):
        if key in out:
            out.pop(key)
            notes.append(f"removed {key} (unused when do_sample=False)")
    return out, notes


def write_safetensors(src: Path, dst: Path, keep: dict, data_start: int,
                      quiet: bool = False) -> int:
    """Stream tensor bytes into a new file with recomputed, contiguous offsets.

    safetensors requires offsets to tile the data section with no gaps, so
    tensors are packed back-to-back in a deterministic (sorted) order.
    """
    ordered = sorted(keep)
    new_header: dict[str, dict] = {}
    offset = 0
    for key in ordered:
        info = keep[key]
        start, end = info["data_offsets"]
        size = end - start
        new_header[key] = {
            "dtype": info["dtype"],
            "shape": info["shape"],
            "data_offsets": [offset, offset + size],
        }
        offset += size
    total = offset

    blob = json.dumps({"__metadata__": {"format": "pt"}, **new_header},
                      separators=(",", ":")).encode("utf-8")
    blob += b" " * (-(8 + len(blob)) % 8)  # align the data section to 8 bytes

    # Carriage-return progress is only meaningful on a terminal; when piped to a
    # log it produces one line per chunk.
    show_progress = not quiet and sys.stdout.isatty()
    written = 0
    t0 = time.time()
    with src.open("rb") as fin, dst.open("wb") as fout:
        fout.write(struct.pack("<Q", len(blob)))
        fout.write(blob)
        for key in ordered:
            start, end = keep[key]["data_offsets"]
            fin.seek(data_start + start)
            remaining = end - start
            while remaining:
                chunk = fin.read(min(remaining, COPY_CHUNK))
                if not chunk:
                    raise OSError(f"unexpected EOF reading {key}")
                fout.write(chunk)
                remaining -= len(chunk)
                written += len(chunk)
            if show_progress and total:
                pct = 100 * written / total
                rate = written / max(time.time() - t0, 1e-6) / GIB
                print(f"\r  copying {pct:5.1f}%  ({written / GIB:.2f}/{total / GIB:.2f} GiB, "
                      f"{rate:.2f} GiB/s)", end="", flush=True)
    if show_progress:
        print()
    elif not quiet:
        elapsed = time.time() - t0
        print(f"  copied {total / GIB:.2f} GiB in {elapsed:.0f}s "
              f"({total / max(elapsed, 1e-6) / GIB:.2f} GiB/s)")
    return total


def verify(path: Path, expected: dict) -> list[str]:
    """Re-read the written file and check structure, offsets and dtype sizes."""
    problems: list[str] = []
    tensors, _, data_start = read_header(path)

    if len(tensors) != len(expected):
        problems.append(f"tensor count {len(tensors)} != expected {len(expected)}")

    cursor = 0
    for key in sorted(tensors):
        info = tensors[key]
        start, end = info["data_offsets"]
        if start != cursor:
            problems.append(f"{key}: gap or overlap at offset {start} (expected {cursor})")
        cursor = end

        elem = DTYPE_SIZES.get(info["dtype"])
        if elem:
            n = 1
            for dim in info["shape"]:
                n *= dim
            if n * elem != end - start:
                problems.append(f"{key}: byte length {end - start} != shape*dtype {n * elem}")

    actual = path.stat().st_size
    if actual != data_start + cursor:
        problems.append(f"file size {actual} != header+data {data_start + cursor}")

    for key in ("model.embed_tokens.weight", "model.norm.weight", "model.layers.0.mlp.up_proj.weight"):
        if key not in tensors:
            problems.append(f"expected key missing from output: {key}")

    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="AI server/gemma4b", help="source checkpoint directory")
    ap.add_argument("--dst", default="AI server/gemma4b-text", help="output directory")
    ap.add_argument("--dry-run", action="store_true", help="report the plan, write nothing")
    ap.add_argument("--force", action="store_true", help="overwrite a non-empty output directory")
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass

    src = Path(args.src)
    dst = Path(args.dst)
    weights = src / "model.safetensors"

    if not weights.is_file():
        print(f"ERROR: {weights} not found", file=sys.stderr)
        return 2

    tensors, metadata, data_start = read_header(weights)
    keep, vision, unknown = classify(tensors)

    kept_bytes = sum(i["data_offsets"][1] - i["data_offsets"][0] for i in keep.values())
    vision_bytes = sum(tensors[k]["data_offsets"][1] - tensors[k]["data_offsets"][0]
                       for k in vision)

    print(f"Source : {src}")
    print(f"  metadata     : {metadata}")
    print(f"  tensors      : {len(tensors)}")
    print(f"  keep (text)  : {len(keep):>4}  {kept_bytes / GIB:6.2f} GiB")
    print(f"  drop (vision): {len(vision):>4}  {vision_bytes / GIB:6.2f} GiB")
    if unknown:
        print(f"  UNEXPECTED   : {len(unknown)} keys matched no rule:")
        for key in unknown[:10]:
            print(f"                 {key}")
        print("  Refusing to continue - the layout is not what this tool expects.")
        return 3
    if not keep:
        print(f"  ERROR: no keys start with {OLD_PREFIX!r}. Already repacked?")
        return 3

    print(f"\nRewrite rule : {OLD_PREFIX!r} -> {NEW_PREFIX!r}")
    sample = sorted(keep)[0]
    original = next(k for k in tensors if k.endswith(sample[len(NEW_PREFIX):]))
    print(f"  example      : {original}")
    print(f"               -> {sample}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return 0

    if dst.exists() and any(dst.iterdir()) and not args.force:
        print(f"\nERROR: {dst} exists and is not empty. Use --force to overwrite.",
              file=sys.stderr)
        return 4
    dst.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting {dst}")
    total = write_safetensors(weights, dst / "model.safetensors", keep, data_start)

    src_cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    (dst / "config.json").write_text(
        json.dumps(build_config(src_cfg), indent=2) + "\n", encoding="utf-8")
    print("  config.json  : Gemma3ForCausalLM / gemma3_text")

    for name in SIDECAR_FILES:
        if (src / name).is_file():
            shutil.copy2(src / name, dst / name)

    gen_path = dst / "generation_config.json"
    if gen_path.is_file():
        fixed, notes = fix_generation_config(
            json.loads(gen_path.read_text(encoding="utf-8")))
        if notes:
            gen_path.write_text(json.dumps(fixed, indent=2) + "\n", encoding="utf-8")
            for note in notes:
                print(f"  generation   : {note}")

    # Inline the chat template so GGUF conversion and any loader pick it up.
    # It lives in a standalone .jinja here, which converters do not always read.
    template = src / "chat_template.jinja"
    if template.is_file():
        shutil.copy2(template, dst / "chat_template.jinja")
        tok_path = dst / "tokenizer_config.json"
        if tok_path.is_file():
            tok_cfg = json.loads(tok_path.read_text(encoding="utf-8"))
            if "chat_template" not in tok_cfg:
                tok_cfg["chat_template"] = template.read_text(encoding="utf-8")
                tok_path.write_text(json.dumps(tok_cfg, indent=2) + "\n", encoding="utf-8")
                print("  tokenizer    : chat template inlined into tokenizer_config.json")

    print("\nVerifying")
    problems = verify(dst / "model.safetensors", keep)
    if problems:
        print("  FAILED:")
        for p in problems:
            print(f"    - {p}")
        return 5

    print(f"  {len(keep)} tensors, {total / GIB:.2f} GiB, offsets contiguous, dtypes consistent")
    print(f"  saved {vision_bytes / GIB:.2f} GiB by dropping the unused vision tower")
    print(f"\nDone. Point SALEDEED_MODEL_MEDIUM_DIR at {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

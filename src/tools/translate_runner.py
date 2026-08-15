"""NLLB-200 translation runner - executed inside the OCR virtual environment.

Shares Surya's interpreter rather than getting its own. Both need torch and
transformers, and a second environment would duplicate ~3 GB of torch for a
600M-parameter model. The subprocess boundary already makes splitting them a
configuration change rather than a rewrite, if the pins ever diverge.

    <venv>/python.exe tools/translate_runner.py --model DIR --in FILE --out FILE

Input and output are JSON. Each item carries its **own** source language,
because a deed is a mixed document - a Kannada name beside an English address -
and forcing one language on the whole batch would mistranslate the rest:

    {"items": [{"id": "b1.name", "text": "ರಮೇಶ್", "src": "kan_Knda",
                "kind": "transliterate"}]}
    {"results": [{"id": "b1.name", "text": "Ramesh"}], "device": "cpu", ...}

**Batching is by source language.** NLLB sets the source through the tokenizer,
so a batch has to be homogeneous; items are grouped before generation rather
than translated one at a time.

**transliterate vs translate.** A person's name must come across by *sound* -
ರಮೇಶ್ is "Ramesh", and translating a proper noun yields nonsense. An address
must come across by *meaning* - ಮುಖ್ಯ ರಸ್ತೆ is "Main Road", not "Mukhya Raste".
NLLB does not expose a transliteration mode, so names are sent with a shorter
beam and a tight length cap, which keeps the model close to the surface form
instead of paraphrasing a fragment into a sentence. That is a mitigation, not a
guarantee: transliteration quality is the weakest part of this design and is
called out in the documentation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

#: NLLB-200 needs ~2.5 GiB on GPU in fp16. Below this, CPU is the faster of two
#: bad options - and on a 4 GB card the language model already holds most of it.
VRAM_NEEDED = 2.8 * (1024 ** 3)

#: Zero-width joiner and non-joiner are meaningful in Indic conjuncts and are
#: preserved. The rest is PDF and OCR noise.
INVISIBLE = "".join(chr(c) for c in (0x200B, 0x200E, 0x200F, 0x202A, 0x202B,
                                     0x202C, 0x202D, 0x202E, 0xFEFF))


def normalise(text: str) -> str:
    """NFC, strip invisible formatting, collapse whitespace.

    NFC matters: Indic vowel signs decompose several ways, and the tokenizer was
    trained on the composed form. A decomposed string silently tokenises worse.
    """
    text = unicodedata.normalize("NFC", text or "")
    text = text.translate({ord(c): None for c in INVISIBLE})
    return " ".join(text.split()).strip()


def choose_device(requested: str = "auto") -> tuple[str, str]:
    """CPU or CUDA, decided from free VRAM at this moment.

    `torch.cuda.is_available()` answers "is there a GPU", not "is there room on
    it". With llama-server resident on a 4 GB card there usually is not, and a
    naive check would send the model to a GPU with nothing left.
    """
    if requested in ("cpu", "cuda"):
        return requested, f"forced by --device {requested}"
    try:
        import torch
    except ImportError:
        return "cpu", "torch is not installed"
    if not torch.cuda.is_available():
        return "cpu", "no CUDA build of torch, or no usable GPU"
    try:
        free, total = torch.cuda.mem_get_info()
    except Exception:  # noqa: BLE001
        return "cuda", "CUDA available (free VRAM unknown)"
    if free < VRAM_NEEDED:
        return "cpu", (f"only {free / 1024 ** 3:.1f} GiB free of "
                       f"{total / 1024 ** 3:.1f} GiB")
    return "cuda", f"{free / 1024 ** 3:.1f} GiB free of {total / 1024 ** 3:.1f} GiB"


def _weights_present(model_dir: Path) -> bool:
    return bool(list(model_dir.glob("*.safetensors")) or
                list(model_dir.glob("pytorch_model*.bin")))


def main() -> int:
    # Before argparse, not after. `--help` prints this module's docstring,
    # which contains non-ASCII text, and a Windows console defaults to cp1252 -
    # so asking a translation tool for help died with a UnicodeEncodeError
    # instead of printing it. Reconfiguring after `parse_args()` was too late:
    # `--help` never returns from there.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="NLLB model directory")
    parser.add_argument("--in", dest="infile", help="JSON input; stdin if absent")
    parser.add_argument("--out", dest="outfile", help="JSON output; stdout if absent")
    parser.add_argument("--tgt", default="eng_Latn")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--probe", action="store_true",
                        help="report readiness and exit without loading")
    args = parser.parse_args()

    model_dir = Path(args.model)
    device, why = choose_device(args.device)

    if args.probe:
        print(json.dumps({
            "ready": _weights_present(model_dir), "device": device, "reason": why,
            "model_dir": str(model_dir), "model": model_dir.name,
            "detail": "ok" if _weights_present(model_dir)
                      else "no model weights in the directory",
        }))
        return 0

    if not model_dir.is_dir() or not _weights_present(model_dir):
        print(f"error: no model weights in {model_dir}", file=sys.stderr)
        return 3

    raw = (Path(args.infile).read_text(encoding="utf-8") if args.infile
           else sys.stdin.read())
    try:
        items = json.loads(raw).get("items") or []
    except json.JSONDecodeError as exc:
        print(f"error: bad input JSON: {exc}", file=sys.stderr)
        return 2

    if not items:
        payload = {"results": [], "device": device, "seconds": 0.0, "count": 0}
        body = json.dumps(payload)
        (Path(args.outfile).write_text(body, encoding="utf-8")
         if args.outfile else sys.stdout.write(body))
        return 0

    started = time.time()
    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    except ImportError as exc:
        print(f"error: transformers/torch unavailable: {exc}", file=sys.stderr)
        return 4

    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.float16 if device == "cuda" else torch.float32)
        model = model.to(device).eval()
    except Exception as exc:  # noqa: BLE001
        print(f"error: could not load model: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 5

    loaded_at = time.time()

    # Grouped by source language: NLLB takes the source from the tokenizer, so a
    # batch must be homogeneous.
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[item.get("src") or "kan_Knda"].append(item)

    results: list[dict] = []
    try:
        for src_lang, group in groups.items():
            tokenizer = AutoTokenizer.from_pretrained(str(model_dir),
                                                      src_lang=src_lang)
            target_id = tokenizer.convert_tokens_to_ids(args.tgt)

            for start in range(0, len(group), args.batch_size):
                chunk = group[start:start + args.batch_size]
                texts = [normalise(i.get("text") or "") for i in chunk]
                # Names and addresses are short. A tight cap and a modest beam
                # keep the model from padding a two-word name into a sentence,
                # which is its main failure mode on fragments.
                names_only = all(i.get("kind") == "transliterate" for i in chunk)
                batch = tokenizer(texts, padding=True, truncation=True,
                                  max_length=256, return_tensors="pt").to(device)
                with torch.inference_mode():
                    generated = model.generate(
                        **batch,
                        forced_bos_token_id=target_id,
                        num_beams=2 if names_only else 4,
                        max_new_tokens=48 if names_only else 200,
                        length_penalty=0.6 if names_only else 1.0,
                        early_stopping=True)
                decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
                for item, text in zip(chunk, decoded):
                    results.append({
                        "id": item.get("id"), "kind": item.get("kind"),
                        "src": src_lang, "source": item.get("text"),
                        "text": text.strip(),
                    })
    except Exception as exc:  # noqa: BLE001
        print(f"error: translation failed: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return 6

    payload = {
        "results": results, "device": device, "model": model_dir.name,
        "count": len(results),
        "seconds": round(time.time() - started, 2),
        "seconds_load": round(loaded_at - started, 2),
        "languages": {lang: len(group) for lang, group in groups.items()},
    }
    body = json.dumps(payload, ensure_ascii=False)
    if args.outfile:
        Path(args.outfile).write_text(body, encoding="utf-8")
        print(f"{len(results)} field(s) in {payload['seconds']}s on {device} "
              f"({', '.join(payload['languages'])})", file=sys.stderr)
    else:
        sys.stdout.write(body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

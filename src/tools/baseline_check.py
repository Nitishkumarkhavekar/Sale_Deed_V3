"""Measure what quantisation costs, against the BF16/f16 weights.

`profiles.py` has warned since it was written that "Q4_K_M is lossy relative to
your trained weights - measure against the BF16 baseline before production
use", and that a quantised KV cache means "verify exact-copy fields (Aadhaar,
PAN) against the baseline". Nothing ever performed either check. This does.

The method, and why it is shaped this way:

  * **OCR is not repeated.** The text already in `ocr_pages` is the input to
    both runs, so the only variable is the model. Re-running Surya would add
    hours and introduce a second source of difference.
  * **Each model runs in its own `llama-server`**, launched directly here
    rather than through the AI server, so a comparison never disturbs a
    production instance and the two runs cannot contaminate one another's KV
    cache.
  * **Fields are compared, not raw text.** Two models phrasing the same PAN
    identically is what matters; whitespace is not.
  * **Exact-copy fields are judged separately.** A wrong sale consideration is
    bad; a wrong Aadhaar is a different category of wrong, because it silently
    attributes a transaction to another person.

Usage:

    # once per model - the f16 run is slow, it will not fit on a 4 GB card
    python tools/baseline_check.py run --model "<path>/deeds-v6_7-Q4_K_M.gguf" \\
        --limit 10 --out runtime/data/baseline_q4.json
    python tools/baseline_check.py run --model "<path>/deeds-v6_7-f16.gguf" \\
        --limit 10 --out runtime/data/baseline_f16.json

    python tools/baseline_check.py compare \\
        --baseline runtime/data/baseline_f16.json \\
        --candidate runtime/data/baseline_q4.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core import paths  # noqa: E402
from core.db.engine import build_engine, build_session_factory, session_scope  # noqa: E402
from core.db.repositories import UnitOfWork  # noqa: E402

#: Fields the model must reproduce **character for character**. A near-miss on
#: any of these is not a small error: it attributes a transaction to a
#: different person, or reports a different amount to the tax authority.
EXACT_FIELDS = ("aadhaar_number", "pan_card_number", "sale_consideration",
                "registration_fee", "transaction_date")

#: Fields where a difference in wording is a real difference but not a
#: misattribution.
TEXT_FIELDS = ("name", "father_name", "gender", "address", "state",
               "schedule_c_property_address", "registration_office")


# ---------------------------------------------------------------------------
# Running one model over a sample
# ---------------------------------------------------------------------------


def _wait_for(port: int, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health",
                                        timeout=3) as response:
                if response.status == 200:
                    return True
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(2)
    return False


def _serve(model: Path, port: int, gpu_layers: int, ctx: int):
    """Start a private llama-server for one model.

    `gpu_layers` is a parameter rather than a constant because the f16 weights
    are 7.3 GB and cannot sit on a 4 GB card: that run offloads to CPU, which
    is slow but is the only way to obtain the reference answer on this hardware.
    """
    binary = paths.TOOLS_DIR / "llamacpp" / "llama-server.exe"
    if not binary.is_file():
        raise SystemExit(f"llama-server not found at {binary}")
    process = subprocess.Popen(
        [str(binary), "-m", str(model), "--host", "127.0.0.1",
         "--port", str(port), "-c", str(ctx), "-ngl", str(gpu_layers),
         "--parallel", "1", "--no-context-shift"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not _wait_for(port, timeout=900):
        process.terminate()
        raise SystemExit(f"{model.name} did not become ready")
    return process


def _complete(port: int, prompt: str, text: str, max_tokens: int) -> str:
    payload = json.dumps({
        "prompt": f"{prompt}\n\n{text}",
        "n_predict": max_tokens,
        # Zero, as in production: a baseline built on sampled output would
        # measure randomness rather than quantisation.
        "temperature": 0.0,
        "repeat_penalty": 1.0,
    }).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/completion", data=payload,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        return json.loads(response.read()).get("content") or ""


def run(model: Path, limit: int, out: Path, gpu_layers: int, ctx: int) -> int:
    from core.pipeline.stages import fit_to_context
    from core.pipeline.stages import extract_json

    prompt = Path(paths.PROMPT_FILE).read_text(encoding="utf-8")
    sessions = build_session_factory(build_engine())

    samples: list[dict[str, Any]] = []
    with session_scope(sessions) as session:
        uow = UnitOfWork(session)
        for batch in uow.batches.list_paginated(1, 50)[0]:
            for doc in uow.documents.list_for_batch(batch.id, per_page=500)[0]:
                text = uow.ocr.full_text(doc)
                if not text.strip():
                    continue
                samples.append({"document": doc.document_id,
                                "identity": doc.transaction_identity or "",
                                "text": text})
                if len(samples) >= limit:
                    break
            if len(samples) >= limit:
                break

    if not samples:
        raise SystemExit("no documents with OCR text; process a batch first")

    print(f"{model.name}: {len(samples)} document(s), {gpu_layers} GPU layers")
    server = _serve(model, port=8090, gpu_layers=gpu_layers, ctx=ctx)
    results = []
    try:
        for n, sample in enumerate(samples, 1):
            sent, trimmed = fit_to_context(sample["text"])
            started = time.perf_counter()
            raw = _complete(8090, prompt, sent, max_tokens=2048)
            elapsed = time.perf_counter() - started
            parsed = extract_json(raw)
            results.append({
                "document": sample["document"],
                "identity": sample["identity"],
                "trimmed": trimmed,
                "seconds": round(elapsed, 2),
                "parsed_ok": parsed is not None,
                "extraction": parsed,
            })
            print(f"  {n}/{len(samples)} {sample['document'][:28]:<30} "
                  f"{elapsed:6.1f}s  {'ok' if parsed else 'UNPARSEABLE'}")
    finally:
        server.terminate()

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"model": model.name, "results": results},
                              ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


# ---------------------------------------------------------------------------
# Comparing two runs
# ---------------------------------------------------------------------------


def _fields(extraction: dict[str, Any] | None) -> dict[str, str]:
    """Flatten an extraction to `path -> value`, so two runs can be diffed."""
    flat: dict[str, str] = {}
    if not extraction:
        return flat
    for side in ("seller_details", "buyer_details"):
        for index, person in enumerate(extraction.get(side) or []):
            for key, value in (person or {}).items():
                flat[f"{side}[{index}].{key}"] = str(value or "").strip()
    for section in ("property_details", "document_details"):
        for key, value in (extraction.get(section) or {}).items():
            flat[f"{section}.{key}"] = str(value or "").strip()
    return flat


def compare(baseline_path: Path, candidate_path: Path) -> int:
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    by_id = {r["document"]: r for r in candidate["results"]}

    print(f"baseline : {baseline['model']}")
    print(f"candidate: {candidate['model']}\n")

    exact_total = exact_same = text_total = text_same = 0
    exact_faults: list[str] = []
    missing = 0

    for row in baseline["results"]:
        other = by_id.get(row["document"])
        if other is None:
            missing += 1
            continue
        left, right = _fields(row["extraction"]), _fields(other["extraction"])
        for key in sorted(set(left) | set(right)):
            a, b = left.get(key, ""), right.get(key, "")
            leaf = key.rsplit(".", 1)[-1]
            if leaf in EXACT_FIELDS:
                exact_total += 1
                if a == b:
                    exact_same += 1
                else:
                    exact_faults.append(
                        f"  {row['document'][:24]:<26} {key:<34} "
                        f"baseline={a!r} candidate={b!r}")
            elif leaf in TEXT_FIELDS:
                text_total += 1
                text_same += a == b

    def pct(part: int, whole: int) -> str:
        return f"{100.0 * part / whole:.2f}%" if whole else "n/a"

    print("=" * 72)
    print(f"  exact-copy fields : {exact_same}/{exact_total} identical "
          f"({pct(exact_same, exact_total)})")
    print(f"  text fields       : {text_same}/{text_total} identical "
          f"({pct(text_same, text_total)})")
    if missing:
        print(f"  {missing} baseline document(s) absent from the candidate run")
    print("=" * 72)

    if exact_faults:
        print(f"\nEXACT-COPY DISAGREEMENTS ({len(exact_faults)}) - each of these "
              "is a different person or a different amount:\n")
        for line in exact_faults[:40]:
            print(line)
        if len(exact_faults) > 40:
            print(f"  ... and {len(exact_faults) - 40} more")
        # A non-zero exit so this can gate a release.
        return 1

    print("\nNo exact-copy field disagreed. Quantisation is safe for the "
          "identifiers on this sample.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Quantisation accuracy baseline")
    sub = ap.add_subparsers(dest="command", required=True)

    runner = sub.add_parser("run", help="extract a sample with one model")
    runner.add_argument("--model", required=True)
    runner.add_argument("--limit", type=int, default=10)
    runner.add_argument("--out", required=True)
    runner.add_argument("--gpu-layers", type=int, default=35,
                        help="0 forces CPU - required for f16 on a small card")
    runner.add_argument("--ctx", type=int, default=16384)

    comp = sub.add_parser("compare", help="diff two runs")
    comp.add_argument("--baseline", required=True)
    comp.add_argument("--candidate", required=True)

    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass

    if args.command == "run":
        return run(Path(args.model), args.limit, Path(args.out),
                   args.gpu_layers, args.ctx)
    return compare(Path(args.baseline), Path(args.candidate))


if __name__ == "__main__":
    raise SystemExit(main())

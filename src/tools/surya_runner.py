"""Surya OCR runner - executed inside Surya's own virtual environment.

This script is never imported by the application. The pipeline launches it as a
subprocess because Surya pins dependency versions that conflict with the rest of
the project, and because loading its three models into the same process as the
language model would put ~3.2 GB of extra weights in VRAM that cannot be released
until the process exits.

    <surya-venv>/python.exe tools/surya_runner.py --pdf FILE --out FILE

Output format matches what the extraction model was finetuned on:

    ===== PAGE 1 =====
    <spatially reconstructed text>
    ===== PAGE 2 =====
    ...

Layout is reconstructed rather than dumped line-by-line: the training corpus
preserves column alignment through space padding, and a deed's schedule and
signature blocks are laid out in columns. Flattening them changes the input
distribution and loses the association between a label and its value.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

#: Character width the page is reconstructed into. 110 matches the corpus the
#: model was trained on; changing it changes the padding the model sees.
TARGET_COLS = 110

#: Surya's three models need roughly this much VRAM. Below it, CUDA would either
#: fail to allocate or thrash, and CPU is the faster of two bad options.
SURYA_VRAM_NEEDED = 3.2 * (1024 ** 3)


def choose_device(requested: str = "auto") -> tuple[str, str]:
    """Pick the compute device, and say why.

    `auto` uses CUDA only when the card has room *right now*. That check is not
    academic: on a 4 GB laptop the language model already holds ~3.2 GiB, so a
    naive `torch.cuda.is_available()` would send Surya to a GPU with nothing left
    and it would die mid-batch. Free VRAM, not merely presence of a GPU, is the
    right question.
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
    if free < SURYA_VRAM_NEEDED:
        return "cpu", (f"only {free / 1024 ** 3:.1f} GiB free of "
                       f"{total / 1024 ** 3:.1f} GiB - not enough for Surya")
    return "cuda", f"{free / 1024 ** 3:.1f} GiB free of {total / 1024 ** 3:.1f} GiB"


def reconstruct_spatial_layout(prediction, image_width: int,
                               target_cols: int = TARGET_COLS) -> str:
    """Rebuild a page as layout-aligned text.

    Groups detected lines into visual rows by vertical overlap, then pads each
    row horizontally so columns line up. This mirrors `run_ocr.py` in the Surya
    folder, which produced the corpus the model was finetuned on.
    """
    lines = getattr(prediction, "text_lines", None) or []
    if not lines:
        return ""

    scale = target_cols / max(1, image_width)
    ordered = sorted(lines, key=lambda x: x.bbox[1])

    rows: list[list[dict]] = []
    for line in ordered:
        bbox, text = line.bbox, line.text
        placed = False
        for row in rows:
            top = min(item["bbox"][1] for item in row)
            bottom = max(item["bbox"][3] for item in row)
            # A generous tolerance: OCR bounding boxes for the same visual row
            # rarely align exactly, especially across scripts.
            tolerance = max((bottom - top) * 0.35, 10.0)
            midpoint = (bbox[1] + bbox[3]) / 2
            if top - tolerance <= midpoint <= bottom + tolerance:
                row.append({"bbox": bbox, "text": text})
                placed = True
                break
        if not placed:
            rows.append([{"bbox": bbox, "text": text}])

    out: list[str] = []
    for row in sorted(rows, key=lambda r: min(i["bbox"][1] for i in r)):
        line_text = ""
        for item in sorted(row, key=lambda x: x["bbox"][0]):
            column = int(item["bbox"][0] * scale)
            if column > len(line_text):
                line_text += " " * (column - len(line_text))
            line_text += item["text"]
        out.append(line_text.rstrip())
    return "\n".join(out)


def line_boxes(prediction, width: int, height: int) -> list[list]:
    """Return `[x0, y0, x1, y1, text]` per detected line, normalised to 0..1.

    Fractions, not pixels: the caller places this text onto a PDF page measured
    in points, and it has no way to know the DPI this process rendered at. A
    normalised box scales by the page rectangle alone, so changing `--dpi` moves
    nothing. Emitted only in `--json` mode; the plain-text output is unchanged.
    """
    out: list[list] = []
    for line in getattr(prediction, "text_lines", None) or []:
        text = (line.text or "").strip()
        if not text:
            continue
        x0, y0, x1, y1 = line.bbox
        out.append([round(x0 / max(1, width), 5), round(y0 / max(1, height), 5),
                    round(x1 / max(1, width), 5), round(y1 / max(1, height), 5),
                    text])
    return out


#: Surya sizes its batches for a workstation card: 32 images for detection and
#: 256 for recognition. On 4 GiB that asks for a single 2.58 GiB tensor and dies
#: with `free: 0` - and it dies harder the longer the document, because the batch
#: follows the page count. A 5-page deed worked and a 14-page deed did not, which
#: is how this hid for so long (R-036).
#:
#: These are per-card ceilings, not tuning. Each is the largest value measured to
#: complete a 14-page deed on that much VRAM.
BATCH_CEILINGS = (
    # (VRAM bytes at least, detector batch, recognition batch)
    (10 * 1024 ** 3, 32, 256),   # workstation - Surya's own defaults
    (6 * 1024 ** 3, 16, 64),
    (0, 4, 16),                  # 4 GiB laptop cards
)


def cap_batches(device: str) -> None:
    """Bound Surya's batch sizes to what this card can actually hold.

    Set through the environment because `surya.settings` is a pydantic model
    read at import time; assigning to it afterwards changes nothing.
    """
    if device != "cuda":
        return                      # CPU defaults are already small (2 and 8)
    try:
        import torch

        total = torch.cuda.get_device_properties(0).total_memory
    except Exception:  # noqa: BLE001
        total = 0

    for floor, detector, recognition in BATCH_CEILINGS:
        if total >= floor:
            break
    else:                           # pragma: no cover - the last rung has floor 0
        detector, recognition = 4, 16

    # Never raise what the operator set deliberately.
    os.environ.setdefault("DETECTOR_BATCH_SIZE", str(detector))
    os.environ.setdefault("RECOGNITION_BATCH_SIZE", str(recognition))
    print(f"batch sizes: detector={os.environ['DETECTOR_BATCH_SIZE']} "
          f"recognition={os.environ['RECOGNITION_BATCH_SIZE']} "
          f"({total / 1024 ** 3:.1f} GiB card)", file=sys.stderr)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", help="PDF to read")
    parser.add_argument("--images", help="directory of pre-rendered page images")
    parser.add_argument("--out", help="write text here instead of stdout")
    parser.add_argument("--dpi", type=int, default=0,
                        help="0 = Surya's high-resolution default (corpus setting)")
    parser.add_argument("--langs", default="kn,en")
    parser.add_argument("--json", action="store_true",
                        help="emit JSON with per-page text and timings")
    parser.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"),
                        help="auto uses CUDA only when enough VRAM is free")
    parser.add_argument("--probe", action="store_true",
                        help="report the device that would be used, then exit")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass

    # Must be decided and exported before Surya is imported: `surya.settings`
    # reads TORCH_DEVICE at import time, and setting it afterwards has no effect.
    device, why = choose_device(args.device)
    os.environ["TORCH_DEVICE"] = device
    cap_batches(device)

    if args.probe:
        payload = {"device": device, "reason": why}
        try:
            import torch

            payload["torch"] = torch.__version__
            payload["cuda_build"] = torch.version.cuda
        except ImportError:
            payload["torch"] = None
        print(json.dumps(payload))
        return 0

    if not args.pdf and not args.images:
        print("error: --pdf or --images is required", file=sys.stderr)
        return 2

    print(f"device: {device} ({why})", file=sys.stderr)
    started = time.time()

    try:
        from surya.detection import DetectionPredictor
        from surya.foundation import FoundationPredictor
        from surya.recognition import RecognitionPredictor
    except ImportError as exc:
        print(f"error: surya is not installed in this interpreter: {exc}",
              file=sys.stderr)
        return 3

    # -- load pages ------------------------------------------------------
    images: list = []
    if args.pdf:
        try:
            from surya.input.load import load_pdf
            from surya.settings import settings

            # IMAGE_DPI_HIGHRES, not the 96 dpi default. This is what `run_ocr.py`
            # used to produce the corpus the extraction model was finetuned on;
            # rendering at a lower resolution changes both the recognised text and
            # the bounding boxes the layout reconstruction depends on.
            dpi = args.dpi or getattr(settings, "IMAGE_DPI_HIGHRES", 192)
            loaded = load_pdf(args.pdf, dpi=dpi)
            # The return arity has changed across Surya releases (images, names)
            # in 0.17; earlier builds appended page numbers. Take the first
            # element rather than unpacking a fixed shape.
            images = loaded[0] if isinstance(loaded, tuple) else loaded
        except Exception as exc:  # noqa: BLE001
            print(f"error: could not read {args.pdf}: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
            return 4
    else:
        from PIL import Image

        for path in sorted(Path(args.images).glob("*.png")):
            images.append(Image.open(path))

    if not images:
        print("error: no pages to process", file=sys.stderr)
        return 5

    loaded_at = time.time()

    # -- run OCR ----------------------------------------------------------
    try:
        foundation = FoundationPredictor()
        recognition = RecognitionPredictor(foundation)
        detection = DetectionPredictor()
        predictions = recognition(images, det_predictor=detection)
    except Exception as exc:  # noqa: BLE001
        print(f"error: OCR failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 6

    # -- assemble ---------------------------------------------------------
    pages: list[str] = []
    chunks: list[str] = []
    boxes: list[list] = []
    for number, (prediction, image) in enumerate(zip(predictions, images), start=1):
        text = reconstruct_spatial_layout(prediction, image.width)
        pages.append(text)
        if args.json:
            boxes.append(line_boxes(prediction, image.width, image.height))
        chunks.append(f"===== PAGE {number} =====")
        chunks.append(text)

    body = "\n".join(chunks) + "\n"
    elapsed = time.time() - started

    if args.json:
        payload = {
            "pages": pages, "page_count": len(pages), "text": body,
            "seconds_total": round(elapsed, 2),
            "seconds_load": round(loaded_at - started, 2),
            "chars": len(body), "device": device,
            # Positions for the invisible text layer that makes a scanned deed
            # searchable. Carried here because they exist only inside this
            # process - the parent sees the subprocess boundary, not Surya.
            "lines": boxes,
        }
        body_out = json.dumps(payload, ensure_ascii=False)
    else:
        body_out = body

    if args.out:
        # LF endings deliberately: CRLF changes tokenisation for the extraction
        # model, and the cleanup stage would only have to undo it.
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(body_out, encoding="utf-8", newline="\n")
        print(f"{len(pages)} page(s), {len(body):,} chars, {elapsed:.1f}s "
              f"on {device} -> {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(body_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

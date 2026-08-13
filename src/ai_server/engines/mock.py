"""Deterministic stub engine - no GPU, no model, no weights.

Exists so the rest of the system (queue, batching, validation, CSV export, UI,
crash recovery) can be built and tested on hardware that cannot host the model,
and so CI can run without a GPU.

It is a *stub*, not a simulator: output is scraped from the OCR text with
regexes rather than inferred. That is deliberate and has a useful property -
every identifier it emits genuinely occurs in the source, so the grounding and
PAN-coverage validators see realistic input and can be exercised end to end.

What it cannot tell you: extraction accuracy. Only the real model measures that.
"""

from __future__ import annotations

import hashlib
import json
import re
import time

from .base import (
    EngineHealth,
    ExtractionRequest,
    ExtractionResult,
    FinishReason,
    InferenceEngine,
)

PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
AADHAAR_RE = re.compile(r"\b(\d{4})[\s-]?(\d{4})[\s-]?(\d{4})\b")
AMOUNT_RE = re.compile(r"(?:rs\.?|₹|inr)\s*([\d,]+(?:\.\d+)?)", re.IGNORECASE)
DATE_RE = re.compile(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})\b")
REG_FEE_RE = re.compile(
    r"(?:registration|regn\.?)\s*(?:fee|fees)|ನೋಂದಣಿ\s*ಶುಲ್ಕ", re.IGNORECASE
)


class MockEngine(InferenceEngine):
    """Returns schema-shaped JSON assembled from identifiers found in the OCR."""

    name = "mock"

    def __init__(self, *, latency_s: float = 0.0, fail_every: int = 0) -> None:
        #: Simulated per-request latency, for exercising queue and progress code.
        self.latency_s = latency_s
        #: Fail every Nth request (0 disables) to exercise retry and recovery.
        self.fail_every = fail_every
        self._started = False
        self._served = 0
        self._last_activity = time.monotonic()

    # -- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        self._started = False

    def is_ready(self) -> bool:
        return self._started

    def health(self) -> EngineHealth:
        return EngineHealth(
            ready=self._started,
            engine=self.name,
            model="mock",
            detail="stub engine - output is scraped from OCR, not inferred",
            device="cpu",
            loaded=self._started,
            requests_served=self._served,
            idle_seconds=time.monotonic() - self._last_activity,
        )

    # -- inference --------------------------------------------------------

    def generate(self, request: ExtractionRequest, timeout_s: float = 600.0) -> ExtractionResult:
        if not self._started:
            self.start()
        if self.latency_s:
            time.sleep(self.latency_s)

        self._served += 1
        self._last_activity = time.monotonic()

        if self.fail_every and self._served % self.fail_every == 0:
            return ExtractionResult(
                text="{ this is deliberately malformed",
                document_id=request.document_id,
                finish_reason=FinishReason.LENGTH,
                engine=self.name,
                model="mock",
                metadata={"injected_failure": True},
            )

        payload = self._scrape(request.ocr_text)
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return ExtractionResult(
            text=text,
            document_id=request.document_id,
            finish_reason=FinishReason.STOP,
            prompt_tokens=len(request.ocr_text) // 4,
            completion_tokens=len(text) // 4,
            duration_s=self.latency_s,
            engine=self.name,
            model="mock",
        )

    # -- scraping ---------------------------------------------------------

    def _scrape(self, ocr: str) -> dict[str, object]:
        pans = list(dict.fromkeys(PAN_RE.findall(ocr)))
        aadhaars = list(dict.fromkeys("".join(m) for m in AADHAAR_RE.findall(ocr)))
        amounts = sorted(
            {int(a.replace(",", "").split(".")[0]) for a in AMOUNT_RE.findall(ocr)},
            reverse=True,
        )

        # Split identifiers across the two sides so cross-side validators have
        # something to chew on. Deterministic: same OCR always yields the same
        # split, so regression diffs stay meaningful.
        half = max(1, len(pans) // 2) if pans else 0
        buyers = self._people(pans[:half], aadhaars[:half], "B")
        sellers = self._people(pans[half:], aadhaars[half:], "S")

        consideration = str(amounts[0]) if amounts else None
        reg_fee = None
        if REG_FEE_RE.search(ocr) and len(amounts) > 1:
            # Registration fee is a small fraction of the consideration; pick the
            # largest candidate that is plausibly ~1%.
            plausible = [a for a in amounts if 100 <= a <= 1_000_000]
            reg_fee = str(plausible[0]) if plausible else None

        date = None
        m = DATE_RE.search(ocr)
        if m:
            d, mo, y = m.groups()
            date = f"{y}-{int(mo):02d}-{int(d):02d}"

        return {
            "buyer_details": buyers,
            "seller_details": sellers,
            "property_details": {
                "schedule_c_property_address": None,
                "state": "Karnataka",
                "sale_consideration": consideration,
                "registration_fee": reg_fee,
                "paid_in_cash": "no",
            },
            "document_details": {
                "transaction_date": date,
                "registration_office": None,
            },
        }

    @staticmethod
    def _people(pans: list[str], aadhaars: list[str], side: str) -> list[dict[str, object]]:
        count = max(len(pans), len(aadhaars))
        people: list[dict[str, object]] = []
        for i in range(count):
            pan = pans[i] if i < len(pans) else None
            aadhaar = aadhaars[i] if i < len(aadhaars) else None
            seed = hashlib.sha256(f"{side}{pan}{aadhaar}{i}".encode()).hexdigest()[:6]
            people.append(
                {
                    "name": f"MOCK {side}{i + 1} {seed.upper()}",
                    "gender": None,
                    "father_name": None,
                    "aadhaar_number": aadhaar,
                    "pan_card_number": pan,
                    "address": None,
                    "state": "Karnataka",
                }
            )
        return people

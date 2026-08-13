"""Deployment classification - what this machine can actually run.

The application has to work on machines that differ enormously: a workstation
with a 12 GB card, the 4 GB laptop this was developed on, an office PC with no
NVIDIA GPU at all, and a thin terminal. Assuming a local GPU and a local database
on all of them is why deployment breaks.

So the machine is classified once at startup and the whole deployment shape
follows from that - which artifacts are needed, which quantisation to build,
whether a local database is required, and whether inference runs here or
elsewhere.

    A  FULL        NVIDIA >= 8 GB    everything local, higher-fidelity weights
    B  CONSTRAINED NVIDIA 4-7 GB     everything local, compact weights
    C  CPU_ONLY    no GPU, >=16 GB RAM  local but slow; explicitly warned
    D  THIN_CLIENT anything else     UI only - AI server and database are remote

Class D is the important one. The AI server already speaks HTTP across a network
boundary, so a weak PC becomes a client of a capable one: no CUDA, no model, no
database server, no administrator rights. One good machine serves many poor ones.
"""

from __future__ import annotations

import os
import platform
import shutil
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from .hardware import GIB, HardwareInfo, detect

#: A 4B model at a usable context needs roughly this much VRAM per rung.
VRAM_FULL = 8 * GIB
VRAM_CONSTRAINED = 4 * GIB
#: CPU inference on less than this is not worth offering.
RAM_CPU_MINIMUM = 16 * GIB
#: Below this the machine should not attempt local inference at all.
RAM_THIN_MAXIMUM = 8 * GIB


class DeploymentClass(str, Enum):
    FULL = "A"
    CONSTRAINED = "B"
    CPU_ONLY = "C"
    THIN_CLIENT = "D"

    @property
    def label(self) -> str:
        return {
            "A": "Full (local GPU inference, higher fidelity)",
            "B": "Constrained (local GPU inference, compact weights)",
            "C": "CPU only (local inference, slow)",
            "D": "Thin client (remote AI server and database)",
        }[self.value]

    @property
    def runs_inference_locally(self) -> bool:
        return self is not DeploymentClass.THIN_CLIENT

    @property
    def needs_local_database(self) -> bool:
        """Only the machine that owns the data needs a database server."""
        return self is not DeploymentClass.THIN_CLIENT


@dataclass(frozen=True)
class ClassPlan:
    """What a class implies for setup and runtime."""

    deployment: DeploymentClass
    reason: str
    #: Quantisation to build and serve. None for thin clients.
    quantisation: str | None
    #: Target context, tokens.
    context: int
    #: llama-server slots.
    parallel: int
    #: Approximate disk needed for model artifacts, bytes.
    disk_bytes: int
    warnings: tuple[str, ...] = ()

    @property
    def needs_cuda(self) -> bool:
        return self.deployment in (DeploymentClass.FULL, DeploymentClass.CONSTRAINED)

    @property
    def needs_model(self) -> bool:
        return self.deployment.runs_inference_locally


def classify(hw: HardwareInfo | None = None) -> ClassPlan:
    """Decide what this machine should be. Never raises."""
    hw = hw or detect()
    gpu = hw.primary_gpu
    vram = gpu.total_bytes if gpu else 0
    ram = hw.ram_total_bytes

    if gpu is not None and vram >= VRAM_FULL:
        # Comfortable headroom: prefer accuracy over footprint. Q6_K is
        # effectively indistinguishable from full precision on this workload and
        # still leaves room for a large context and several slots.
        return ClassPlan(
            deployment=DeploymentClass.FULL,
            reason=f"{gpu.name} with {vram / GIB:.0f} GiB VRAM",
            quantisation="Q6_K", context=32768, parallel=4,
            disk_bytes=int(12.5 * GIB))

    if gpu is not None and vram >= VRAM_CONSTRAINED:
        return ClassPlan(
            deployment=DeploymentClass.CONSTRAINED,
            reason=f"{gpu.name} with {vram / GIB:.1f} GiB VRAM",
            quantisation="Q4_K_M", context=24576, parallel=1,
            disk_bytes=int(11 * GIB),
            warnings=(
                "4-bit weights measurably damage exact-digit fields; validation "
                "cross-checks every extracted value against the OCR source.",
            ))

    if ram >= RAM_CPU_MINIMUM:
        return ClassPlan(
            deployment=DeploymentClass.CPU_ONLY,
            reason=(f"no usable NVIDIA GPU, {ram / GIB:.0f} GiB RAM"
                    if gpu is None else
                    f"{gpu.name} has only {vram / GIB:.1f} GiB VRAM"),
            quantisation="Q4_K_M", context=8192, parallel=1,
            disk_bytes=int(11 * GIB),
            warnings=(
                "CPU inference is roughly an order of magnitude slower than GPU. "
                "Expect minutes per document, not seconds.",
                "Context is reduced to 8192 tokens, below the corpus median of "
                "9.4k - long deeds will be rejected rather than truncated.",
            ))

    return ClassPlan(
        deployment=DeploymentClass.THIN_CLIENT,
        reason=(f"no usable NVIDIA GPU and only {ram / GIB:.1f} GiB RAM"
                if gpu is None else
                f"{vram / GIB:.1f} GiB VRAM and {ram / GIB:.1f} GiB RAM"),
        quantisation=None, context=0, parallel=0, disk_bytes=0,
        warnings=(
            "This machine will run the interface only. Set SALEDEED_AI_URL and "
            "SALEDEED_DB_URL to a machine that hosts them.",
        ))


# ---------------------------------------------------------------------------
# Prerequisites
# ---------------------------------------------------------------------------


@dataclass
class Requirement:
    name: str
    present: bool
    detail: str = ""
    fix: str = ""
    optional: bool = False

    @property
    def status(self) -> str:
        if self.present:
            return "ok"
        return "optional" if self.optional else "MISSING"


@dataclass
class Readiness:
    plan: ClassPlan
    hardware: HardwareInfo
    requirements: list[Requirement] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return all(r.present for r in self.requirements if not r.optional)

    @property
    def missing(self) -> list[Requirement]:
        return [r for r in self.requirements if not r.present and not r.optional]

    def report(self) -> str:
        lines = [
            "Deployment assessment",
            f"  Machine   : {self.hardware.cpu_name}",
            f"  Class     : {self.plan.deployment.value} - {self.plan.deployment.label}",
            f"  Reason    : {self.plan.reason}",
        ]
        if self.plan.quantisation:
            lines.append(f"  Model     : {self.plan.quantisation}, "
                         f"{self.plan.context:,} context, {self.plan.parallel} slot(s)")
            lines.append(f"  Disk      : ~{self.plan.disk_bytes / GIB:.1f} GiB for artifacts")
        lines.append("")
        lines.append("Prerequisites")
        for req in self.requirements:
            mark = {"ok": "  [ok]  ", "MISSING": "  [ -- ]", "optional": "  [opt] "}[req.status]
            lines.append(f"{mark} {req.name:<26} {req.detail}")
            if not req.present and req.fix:
                lines.append(f"         -> {req.fix}")
        for warning in self.plan.warnings:
            lines.append(f"\n  NOTE: {warning}")
        return "\n".join(lines)


def _python_ok() -> Requirement:
    version = ".".join(str(p) for p in sys.version_info[:3])
    ok = sys.version_info >= (3, 12)
    return Requirement("Python >= 3.12", ok, version,
                       "install Python 3.12 or newer from python.org")


def _package(module: str, label: str, hint: str, optional: bool = False) -> Requirement:
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "installed")
        return Requirement(label, True, str(version), optional=optional)
    except Exception:  # noqa: BLE001
        return Requirement(label, False, "not installed", hint, optional=optional)


def _llama_server(root: Path) -> Requirement:
    local = root / "tools" / "llamacpp" / "llama-server.exe"
    found = str(local) if local.is_file() else shutil.which("llama-server")
    if not found:
        return Requirement(
            "llama-server (CUDA)", False, "not found",
            "python tools/setup.py --install-runtime")
    cudart = (root / "tools" / "llamacpp" / "cudart64_12.dll").is_file()
    return Requirement("llama-server (CUDA)", True,
                       f"{found}{'' if cudart else '  (CUDA runtime DLLs missing)'}")


def _model(root: Path, quantisation: str | None) -> Requirement:
    if quantisation is None:
        return Requirement("GGUF model", True, "not needed for a thin client",
                           optional=True)
    target = root.parent / "models" / "AI server" / "gguf" / f"deeds-v6_7-{quantisation}.gguf"
    if target.is_file():
        return Requirement(f"GGUF model ({quantisation})", True,
                           f"{target.stat().st_size / GIB:.2f} GiB")
    source = root.parent / "models" / "AI server" / "gemma4b"
    if not (source / "model.safetensors").is_file():
        return Requirement(f"GGUF model ({quantisation})", False,
                           "trained checkpoint not found",
                           f"place the trained model in {source}")
    return Requirement(f"GGUF model ({quantisation})", False, "not built",
                       f"python tools/setup.py --build-model --quant {quantisation}")


def _database(plan: ClassPlan) -> Requirement:
    dsn = os.environ.get("SALEDEED_DB_URL", "")
    if not plan.deployment.needs_local_database:
        ok = bool(dsn) and "localhost" not in dsn and "127.0.0.1" not in dsn
        return Requirement(
            "Database (remote)", ok,
            dsn.split("@")[-1] if dsn else "SALEDEED_DB_URL not set",
            "set SALEDEED_DB_URL to the host machine's PostgreSQL")

    from .hardware import _run  # local import: only needed here

    binary = shutil.which("psql") or _postgres_bin()
    if binary is None:
        return Requirement("PostgreSQL server", False, "not installed",
                           "python tools/setup.py --install-database")
    return Requirement("PostgreSQL server", True, binary)


def _postgres_bin() -> str | None:
    # %ProgramFiles% rather than a literal C:. The folder is not always on C:,
    # and on a localized Windows it is not always called "Program Files".
    roots = {os.environ.get("ProgramFiles", r"C:\Program Files"),
             os.environ.get("ProgramW6432", r"C:\Program Files"),
             os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")}
    for version in ("18", "17", "16", "15"):
        for root in roots:
            candidate = Path(root) / "PostgreSQL" / version / "bin" / "psql.exe"
            if candidate.is_file():
                return str(candidate)
    return None


def _ai_server(plan: ClassPlan) -> Requirement:
    url = os.environ.get("SALEDEED_AI_URL", "http://127.0.0.1:8077")
    if plan.deployment.runs_inference_locally:
        return Requirement("AI server", True, f"local, {url}", optional=True)
    remote = "127.0.0.1" not in url and "localhost" not in url
    return Requirement("AI server (remote)", remote, url,
                       "set SALEDEED_AI_URL to the host machine")


def _disk(root: Path, plan: ClassPlan) -> Requirement:
    try:
        free = shutil.disk_usage(root).free
    except OSError:
        return Requirement("Disk space", True, "unknown", optional=True)
    needed = plan.disk_bytes
    ok = free >= needed
    return Requirement("Disk space", ok,
                       f"{free / GIB:.0f} GiB free, ~{needed / GIB:.1f} GiB needed",
                       "free space or move the project to a larger volume")


def _driver(hw: HardwareInfo, plan: ClassPlan) -> Requirement:
    if not plan.needs_cuda:
        return Requirement("NVIDIA driver", True, "not required", optional=True)
    if not hw.driver_version:
        return Requirement("NVIDIA driver", False, "not detected",
                           "install the NVIDIA driver from nvidia.com")
    try:
        major = int(str(hw.driver_version).split(".")[0])
    except ValueError:
        major = 0
    # The CUDA 12.4 build needs roughly 527+; anything older cannot load it.
    ok = major >= 527
    return Requirement("NVIDIA driver", ok,
                       f"{hw.driver_version} (CUDA {hw.cuda_version or '?'})",
                       "update the NVIDIA driver to 527 or newer")


def assess(root: Path | None = None, hw: HardwareInfo | None = None) -> Readiness:
    """Classify the machine and check everything that class needs."""
    base = root or Path(__file__).resolve().parents[1]
    hardware = hw or detect(disk_paths=(str(base),))
    plan = classify(hardware)

    requirements = [
        _python_ok(),
        _package("PySide6", "PySide6 (desktop UI)", "pip install PySide6"),
        _package("pystache", "Pystache (templates)", "pip install pystache"),
        _package("sqlalchemy", "SQLAlchemy", "pip install SQLAlchemy"),
        _package("psycopg", "psycopg (v3)", "pip install psycopg[binary]"),
        _package("alembic", "Alembic", "pip install alembic"),
        _package("pymupdf", "PyMuPDF (PDF)", "pip install PyMuPDF"),
        _database(plan),
        _ai_server(plan),
    ]
    if plan.needs_cuda:
        requirements.append(_driver(hardware, plan))
        requirements.append(_llama_server(base))
    if plan.needs_model:
        requirements.append(_model(base, plan.quantisation))
        requirements.append(_disk(base, plan))

    return Readiness(plan=plan, hardware=hardware, requirements=requirements)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass
    readiness = assess()
    print(readiness.report())
    print()
    print("READY" if readiness.ready else
          f"NOT READY - {len(readiness.missing)} prerequisite(s) missing")
    raise SystemExit(0 if readiness.ready else 1)

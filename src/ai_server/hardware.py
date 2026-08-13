"""Hardware detection for the Sale Deed AI server.

Standard library only, on purpose: profile selection must work on a bare
interpreter, before torch or any inference runtime is installed, so the
installer can decide *what* to install based on what it finds.

Rules enforced here, straight from the deployment requirements:

  * Only a dedicated NVIDIA GPU is ever used for inference.
  * The AMD integrated GPU is detected purely so it can be excluded, and so we
    can warn loudly. CUDA cannot see it, but a Vulkan or OpenCL build of
    llama.cpp would enumerate it and may silently select it - which would be
    catastrophically slow and hard to notice.
  * GPUs are pinned by UUID, never by index: indices reorder across driver
    updates and reboots, UUIDs do not.
  * CPU is an explicit fallback, never a silent default.
"""

from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
from dataclasses import dataclass, field

MIB = 1024 * 1024
GIB = 1024 * 1024 * 1024

# nvidia-smi occasionally hangs on a wedged driver; never block startup on it.
_SMI_TIMEOUT_S = 15


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GpuDevice:
    """A single CUDA-capable device."""

    index: int
    uuid: str
    name: str
    total_bytes: int
    free_bytes: int
    used_bytes: int
    compute_capability: str | None

    @property
    def total_gib(self) -> float:
        return self.total_bytes / GIB

    @property
    def free_gib(self) -> float:
        return self.free_bytes / GIB

    @property
    def is_display_gpu(self) -> bool:
        """Heuristic: a GPU already holding memory is probably driving a display.

        Matters because a display GPU permanently loses ~0.5-1.0 GB to the
        desktop compositor, which has to come out of the inference budget.
        """
        return self.used_bytes > 256 * MIB

    def __str__(self) -> str:
        cc = f" sm_{self.compute_capability.replace('.', '')}" if self.compute_capability else ""
        return (
            f"[{self.index}] {self.name}{cc} - "
            f"{self.free_gib:.2f} GiB free of {self.total_gib:.2f} GiB"
        )


@dataclass(frozen=True)
class DiskInfo:
    """Free space on a volume the application writes to."""

    path: str
    total_bytes: int
    free_bytes: int

    @property
    def free_gib(self) -> float:
        return self.free_bytes / GIB

    @property
    def used_fraction(self) -> float:
        return 1.0 - (self.free_bytes / self.total_bytes) if self.total_bytes else 0.0


@dataclass(frozen=True)
class HardwareInfo:
    """Everything profile selection needs to know about this machine."""

    os_name: str
    cpu_name: str
    logical_cores: int
    physical_cores: int
    ram_total_bytes: int
    ram_available_bytes: int

    cuda_available: bool
    driver_version: str | None
    cuda_version: str | None
    gpus: list[GpuDevice] = field(default_factory=list)
    disks: list[DiskInfo] = field(default_factory=list)

    # Non-NVIDIA adapters, recorded only so they can be excluded.
    other_adapters: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    # -- convenience ------------------------------------------------------

    @property
    def ram_total_gib(self) -> float:
        return self.ram_total_bytes / GIB

    @property
    def ram_available_gib(self) -> float:
        return self.ram_available_bytes / GIB

    @property
    def ram_pressure(self) -> float:
        """Fraction of RAM in use. Above ~0.9 the machine is swapping."""
        if not self.ram_total_bytes:
            return 0.0
        return 1.0 - (self.ram_available_bytes / self.ram_total_bytes)

    @property
    def primary_gpu(self) -> GpuDevice | None:
        """The NVIDIA device inference should use: most free VRAM wins."""
        if not self.gpus:
            return None
        return max(self.gpus, key=lambda g: g.free_bytes)

    def cuda_visible_devices(self) -> str | None:
        """Value for CUDA_VISIBLE_DEVICES pinning inference to one GPU.

        Returns the UUID rather than the index deliberately - CUDA accepts
        `GPU-<uuid>` and it survives driver updates that renumber devices.
        """
        gpu = self.primary_gpu
        return gpu.uuid if gpu else None

    def summary(self) -> str:
        lines = [
            "Hardware",
            f"  OS            : {self.os_name}",
            f"  CPU           : {self.cpu_name}",
            f"  Cores         : {self.physical_cores} physical / {self.logical_cores} logical",
            f"  RAM           : {self.ram_available_gib:.1f} GiB available "
            f"of {self.ram_total_gib:.1f} GiB ({self.ram_pressure:.0%} in use)",
        ]
        for disk in self.disks:
            lines.append(
                f"  Disk {disk.path:<9}: {disk.free_gib:.1f} GiB free "
                f"({disk.used_fraction:.0%} used)"
            )
        if self.cuda_available:
            lines.append(
                f"  Driver / CUDA : {self.driver_version or '?'} / {self.cuda_version or '?'}"
            )
            for gpu in self.gpus:
                marker = " <- selected" if gpu is self.primary_gpu else ""
                lines.append(f"  GPU           : {gpu}{marker}")
                if gpu.is_display_gpu:
                    lines.append(
                        f"                  (driving a display: "
                        f"{gpu.used_bytes / MIB:.0f} MiB already in use)"
                    )
        else:
            lines.append("  GPU           : no CUDA device - CPU fallback")
        for adapter in self.other_adapters:
            lines.append(f"  Excluded      : {adapter} (never used for inference)")
        for w in self.warnings:
            lines.append(f"  WARNING       : {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# System memory
# ---------------------------------------------------------------------------


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


def _system_memory() -> tuple[int, int]:
    """Return (total_bytes, available_bytes) without requiring psutil."""
    if os.name == "nt":
        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.ullTotalPhys), int(status.ullAvailPhys)
        return 0, 0

    try:  # Linux
        meminfo: dict[str, int] = {}
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, rest = line.partition(":")
                meminfo[key.strip()] = int(rest.strip().split()[0]) * 1024
        total = meminfo.get("MemTotal", 0)
        avail = meminfo.get("MemAvailable", meminfo.get("MemFree", 0))
        return total, avail
    except (OSError, ValueError, IndexError):
        pass

    try:  # macOS / BSD
        total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                   text=True, timeout=5, check=True).stdout.strip())
        return total, total // 2  # no cheap availability figure; assume half
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0, 0


def _cpu_cores() -> tuple[int, int]:
    """Return (physical, logical) cores.

    Physical count drives the llama.cpp thread setting: SMT siblings actively
    hurt throughput for memory-bound GEMMs, so we want real cores only.
    """
    logical = os.cpu_count() or 1

    if os.name == "nt":
        cores = os.environ.get("NUMBER_OF_PHYSICAL_PROCESSORS")
        if cores and cores.isdigit():
            return int(cores), logical
        # Assume SMT. True for every Ryzen/Core part this app targets.
        return max(1, logical // 2), logical

    try:  # Linux: count distinct (physical id, core id) pairs
        pairs: set[tuple[str, str]] = set()
        phys = core = None
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                key, _, val = line.partition(":")
                key, val = key.strip(), val.strip()
                if key == "physical id":
                    phys = val
                elif key == "core id":
                    core = val
                elif not line.strip() and phys and core:
                    pairs.add((phys, core))
                    phys = core = None
        if pairs:
            return len(pairs), logical
    except OSError:
        pass

    return max(1, logical // 2), logical


def _cpu_name() -> str:
    if os.name == "nt":
        name = os.environ.get("PROCESSOR_IDENTIFIER", "")
        if name:
            return name.strip()
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.partition(":")[2].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


# ---------------------------------------------------------------------------
# NVIDIA detection
# ---------------------------------------------------------------------------


def _run(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_SMI_TIMEOUT_S, check=False
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _nvidia_smi_path() -> str | None:
    found = shutil.which("nvidia-smi")
    if found:
        return found
    # Not on PATH in some service contexts; check the standard install location.
    if os.name == "nt":
        candidate = os.path.join(
            os.environ.get("SystemRoot", r"C:\Windows"), "System32", "nvidia-smi.exe"
        )
        if os.path.isfile(candidate):
            return candidate
    return None


def _query_gpus(smi: str) -> tuple[list[GpuDevice], list[str]]:
    warnings: list[str] = []
    fields = "index,uuid,name,memory.total,memory.used,memory.free,compute_cap"
    out = _run([smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"])

    if out is None:  # older drivers reject compute_cap
        fields = "index,uuid,name,memory.total,memory.used,memory.free"
        out = _run([smi, f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
        if out is None:
            warnings.append("nvidia-smi present but the GPU query failed")
            return [], warnings

    gpus: list[GpuDevice] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            gpus.append(
                GpuDevice(
                    index=int(parts[0]),
                    uuid=parts[1],
                    name=parts[2],
                    total_bytes=int(float(parts[3])) * MIB,
                    used_bytes=int(float(parts[4])) * MIB,
                    free_bytes=int(float(parts[5])) * MIB,
                    compute_capability=parts[6] if len(parts) > 6 else None,
                )
            )
        except (ValueError, IndexError):
            warnings.append(f"could not parse nvidia-smi row: {line!r}")
    return gpus, warnings


def _driver_and_cuda(smi: str) -> tuple[str | None, str | None]:
    out = _run([smi]) or ""
    driver = cuda = None
    m = re.search(r"Driver Version:\s*([\d.]+)", out)
    if m:
        driver = m.group(1)
    m = re.search(r"CUDA Version:\s*([\d.]+)", out)
    if m:
        cuda = m.group(1)
    return driver, cuda


def _other_adapters(nvidia_names: list[str]) -> list[str]:
    """Non-NVIDIA display adapters, recorded so they can be excluded.

    Best effort and never fatal - this is a warning path, not a control path.
    """
    if os.name != "nt":
        return []
    out = _run(
        [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name",
        ]
    )
    if not out:
        return []
    others = []
    for line in out.strip().splitlines():
        name = line.strip()
        if name and "nvidia" not in name.lower() and name not in nvidia_names:
            others.append(name)
    return others


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def disk_for(*paths: str | os.PathLike[str]) -> list[DiskInfo]:
    """Free space on the volumes backing the given paths, de-duplicated.

    Batch processing writes rendered page images, OCR text and the database;
    running a volume dry mid-batch corrupts a run, so this is a preflight input
    rather than a diagnostic.
    """
    seen: dict[str, DiskInfo] = {}
    for raw in paths or (os.getcwd(),):
        path = os.path.abspath(raw)
        # Walk up to the nearest existing ancestor - the target may not exist yet.
        probe = path
        while probe and not os.path.exists(probe):
            parent = os.path.dirname(probe)
            if parent == probe:
                break
            probe = parent
        try:
            usage = shutil.disk_usage(probe)
        except OSError:
            continue
        key = os.path.splitdrive(probe)[0] or probe
        seen.setdefault(key, DiskInfo(path=key, total_bytes=usage.total,
                                      free_bytes=usage.free))
    return list(seen.values())


def detect(
    probe_other_adapters: bool = True,
    disk_paths: tuple[str, ...] = (),
) -> HardwareInfo:
    """Inspect this machine. Never raises; degrades to a CPU-only report."""
    warnings: list[str] = []
    ram_total, ram_avail = _system_memory()
    physical, logical = _cpu_cores()
    disks = disk_for(*disk_paths) if disk_paths else disk_for()

    smi = _nvidia_smi_path()
    gpus: list[GpuDevice] = []
    driver = cuda = None

    if smi:
        gpus, gpu_warnings = _query_gpus(smi)
        warnings.extend(gpu_warnings)
        driver, cuda = _driver_and_cuda(smi)
    else:
        warnings.append(
            "nvidia-smi not found - no NVIDIA GPU usable. Inference will fall back to CPU."
        )

    others = _other_adapters([g.name for g in gpus]) if probe_other_adapters else []
    if others:
        warnings.append(
            "Non-NVIDIA adapter present. Use a CUDA build of the runtime: a Vulkan or "
            "OpenCL build would enumerate it and may select it silently."
        )

    return HardwareInfo(
        os_name=f"{platform.system()} {platform.release()}",
        cpu_name=_cpu_name(),
        logical_cores=logical,
        physical_cores=physical,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_avail,
        cuda_available=bool(gpus),
        driver_version=driver,
        cuda_version=cuda,
        gpus=gpus,
        disks=disks,
        other_adapters=others,
        warnings=warnings,
    )


if __name__ == "__main__":
    print(detect().summary())

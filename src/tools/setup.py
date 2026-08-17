"""One-command setup for any Windows PC.

Detects what this machine is, then installs only what that class needs. Every
step is idempotent - re-running skips work already done - so a failed or
interrupted setup can simply be run again.

    python tools/setup.py                 assess only, change nothing
    python tools/setup.py --all           install everything this class needs
    python tools/setup.py --install-deps
    python tools/setup.py --install-runtime      llama.cpp CUDA binaries
    python tools/setup.py --install-database     PostgreSQL + role + schema
    python tools/setup.py --build-model          repack -> GGUF -> quantise
    python tools/setup.py --thin-client HOST     configure as a remote client

Classes
    A  NVIDIA >= 8 GB   builds Q6_K - near-lossless, large context
    B  NVIDIA 4-7 GB    builds Q4_K_M - compact
    C  no GPU, >=16 GB  builds Q4_K_M, runs on CPU (slow)
    D  anything else    installs the UI only and points at a remote host

The model is never downloaded. Everything is derived from the trained checkpoint
already in `AI server/gemma4b`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: `alembic.ini` sits beside the five top-level folders, not inside `src`, and
#: its `script_location` is written relative to itself. Alembic must therefore be
#: launched from the project root - run from `src` it reports "No 'script_location'
#: key found in configuration" and exits non-zero, which is exactly what every
#: --upgrade path here did until this was noticed while adding a migration.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ai_server.deployment import DeploymentClass, assess  # noqa: E402
from ai_server.hardware import GIB  # noqa: E402
from core import paths

LLAMA_RELEASE = "b10184"
LLAMA_BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_RELEASE}/"
#: CUDA 12.4 rather than 13.x: it runs on driver 527+, which covers far more
#: machines. The 13.x build needs 580+ and would exclude working hardware.
LLAMA_ASSETS = (
    f"llama-{LLAMA_RELEASE}-bin-win-cuda-12.4-x64.zip",
    "cudart-llama-bin-win-cuda-12.4-x64.zip",
)
LLAMA_CPU_ASSET = f"llama-{LLAMA_RELEASE}-bin-win-cpu-x64.zip"

RUNTIME_PACKAGES = (
    "PySide6-Essentials>=6.8", "PySide6-Addons>=6.8", "pystache>=0.6.5",
    "SQLAlchemy>=2.0.36", "psycopg[binary]>=3.2", "alembic>=1.14",
    "PyMuPDF>=1.24",
)
#: Only needed to *build* a GGUF; not required to run the application.
BUILD_PACKAGES = (
    "numpy>=2.1", "sentencepiece>=0.1.98,<0.3.0", "transformers==4.57.6",
    "gguf>=0.1.0", "protobuf>=4.21,<5.0",
)

CONVERT_SRC = ROOT / "tools" / "llamacpp-src"
LLAMA_DIR = ROOT / "tools" / "llamacpp"
GGUF_DIR = paths.GGUF_DIR
CHECKPOINT = paths.CHECKPOINT_DIR
REPACKED = paths.REPACKED_DIR


def _step(title: str) -> None:
    print(f"\n--- {title} ---")


def _ok(msg: str) -> None:
    print(f"  [ok]   {msg}")


def _fail(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def _run(command: list[str], timeout: float = 1800.0, **kwargs) -> tuple[int, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace",
                                timeout=timeout, check=False, **kwargs)
    except subprocess.TimeoutExpired:
        return 1, f"timed out after {timeout:.0f}s"
    except OSError as exc:
        return 1, str(exc)
    return result.returncode, (result.stderr or result.stdout or "").strip()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------


def install_dependencies(build_tools: bool = False) -> bool:
    _step("Python packages")
    packages = list(RUNTIME_PACKAGES) + (list(BUILD_PACKAGES) if build_tools else [])
    code, output = _run([sys.executable, "-m", "pip", "install", "--quiet", *packages])
    if code != 0:
        _fail(output.splitlines()[-1][:200] if output else "pip failed")
        return False
    _ok(f"{len(packages)} package(s) installed or already present")
    return True


def _download(name: str, target: Path) -> bool:
    if target.is_file() and target.stat().st_size > 1024:
        return True
    print(f"  downloading {name} ...", flush=True)
    try:
        urllib.request.urlretrieve(LLAMA_BASE + name, target)
    except Exception as exc:  # noqa: BLE001
        _fail(f"{name}: {exc}")
        return False
    return True


def install_runtime(cpu_only: bool = False) -> bool:
    """Fetch llama.cpp binaries. Skipped entirely if already present."""
    _step("Inference runtime (llama.cpp)")
    if (LLAMA_DIR / "llama-server.exe").is_file() and (
            cpu_only or (LLAMA_DIR / "cudart64_12.dll").is_file()):
        _ok(f"already installed at {LLAMA_DIR}")
        return True

    LLAMA_DIR.mkdir(parents=True, exist_ok=True)
    assets = (LLAMA_CPU_ASSET,) if cpu_only else LLAMA_ASSETS
    for name in assets:
        archive = LLAMA_DIR / name
        if not _download(name, archive):
            return False
        try:
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(LLAMA_DIR)
        except zipfile.BadZipFile:
            archive.unlink(missing_ok=True)
            _fail(f"{name} was corrupt; re-run to download again")
            return False
        archive.unlink(missing_ok=True)

    if not (LLAMA_DIR / "llama-server.exe").is_file():
        _fail("llama-server.exe missing after extraction")
        return False
    _ok(f"installed to {LLAMA_DIR}")
    return True


def install_database(password: str = "saledeed") -> bool:
    """Install PostgreSQL if absent, then create the role, database and schema."""
    _step("PostgreSQL")
    psql = shutil.which("psql") or _find_psql()

    if psql is None:
        print("  installing PostgreSQL 17 (this takes several minutes) ...", flush=True)
        code, output = _run([
            "winget", "install", "--id", "PostgreSQL.PostgreSQL.17",
            "--source", "winget", "--accept-package-agreements",
            "--accept-source-agreements", "--custom",
            f"--mode unattended --unattendedmodeui minimal "
            f"--superpassword {password}_super --serverport 5432 --enable_acledit 1",
        ], timeout=3600)
        if code != 0:
            _fail(f"winget failed: {output.splitlines()[-1][:160] if output else code}")
            print("       Install PostgreSQL manually from postgresql.org, then re-run.")
            return False
        psql = _find_psql()
        if psql is None:
            _fail("PostgreSQL installed but psql.exe not found")
            return False
    _ok(f"psql at {psql}")

    env = dict(os.environ, PGPASSWORD=f"{password}_super")
    role_sql = (f"DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles "
                f"WHERE rolname='saledeed') THEN CREATE ROLE saledeed LOGIN "
                f"PASSWORD '{password}'; END IF; END $$;")
    code, output = _run([psql, "-U", "postgres", "-h", "localhost", "-d", "postgres",
                         "-v", "ON_ERROR_STOP=1", "-c", role_sql], env=env)
    if code != 0:
        _fail(f"could not create role: {output[:160]}")
        return False
    _ok("role 'saledeed' present")

    code, output = _run([psql, "-U", "postgres", "-h", "localhost", "-d", "postgres",
                         "-tAc", "SELECT 1 FROM pg_database WHERE datname='saledeed'"],
                        env=env)
    if output.strip() != "1":
        code, output = _run([psql, "-U", "postgres", "-h", "localhost", "-d", "postgres",
                             "-c", "CREATE DATABASE saledeed OWNER saledeed"], env=env)
        if code != 0:
            _fail(f"could not create database: {output[:160]}")
            return False
    _ok("database 'saledeed' present")

    dsn = f"postgresql+psycopg://saledeed:{password}@localhost:5432/saledeed"
    code, output = _run([sys.executable, "-m", "alembic", "upgrade", "head"],
                        cwd=PROJECT_ROOT, env=dict(os.environ, SALEDEED_DB_URL=dsn))
    if code != 0:
        _fail(f"migration failed: {output.splitlines()[-1][:160] if output else code}")
        return False
    _ok("schema migrated to head")
    print(f"\n  Set this in your environment:\n    SALEDEED_DB_URL={dsn}")
    return True


def _find_psql() -> str | None:
    for version in ("18", "17", "16", "15"):
        candidate = Path(rf"C:\Program Files\PostgreSQL\{version}\bin\psql.exe")
        if candidate.is_file():
            return str(candidate)
    return None


def build_model(quantisation: str) -> bool:
    """Repack the trained checkpoint, convert to GGUF, quantise.

    Nothing is downloaded. Every artifact derives from `AI server/gemma4b`.
    """
    _step(f"Model build ({quantisation})")
    target = GGUF_DIR / f"deeds-v6_7-{quantisation}.gguf"
    if target.is_file():
        _ok(f"{target.name} already built ({target.stat().st_size / GIB:.2f} GiB)")
        return True

    if not (CHECKPOINT / "model.safetensors").is_file():
        _fail(f"trained checkpoint not found at {CHECKPOINT}")
        return False

    # 1. Lossless repack (renames keys, drops the unused vision tower).
    if not (REPACKED / "model.safetensors").is_file():
        print("  repacking checkpoint ...", flush=True)
        code, output = _run([sys.executable, str(ROOT / "tools" / "repack_checkpoint.py")],
                            cwd=ROOT)
        if code != 0:
            _fail(f"repack failed: {output.splitlines()[-1][:160] if output else code}")
            return False
    _ok("repacked checkpoint present")

    # 2. f16 GGUF - the shared intermediate for every quantisation.
    f16 = GGUF_DIR / "deeds-v6_7-f16.gguf"
    if not f16.is_file():
        if not _ensure_converter():
            return False
        GGUF_DIR.mkdir(parents=True, exist_ok=True)
        print("  converting to f16 GGUF (several minutes) ...", flush=True)
        code, output = _run([
            sys.executable, str(CONVERT_SRC / "convert_hf_to_gguf.py"), str(REPACKED),
            "--outfile", str(f16), "--outtype", "f16"], timeout=3600)
        if code != 0 or not f16.is_file():
            _fail(f"conversion failed: {output.splitlines()[-1][:200] if output else code}")
            return False
    _ok(f"f16 GGUF present ({f16.stat().st_size / GIB:.2f} GiB)")

    # 3. Quantise.
    quantize = LLAMA_DIR / "llama-quantize.exe"
    if not quantize.is_file():
        _fail("llama-quantize.exe not found - run --install-runtime first")
        return False
    print(f"  quantising to {quantisation} ...", flush=True)
    code, output = _run([str(quantize), str(f16), str(target), quantisation, "8"],
                        timeout=3600)
    if code != 0 or not target.is_file():
        _fail(f"quantisation failed: {output.splitlines()[-1][:200] if output else code}")
        return False
    _ok(f"{target.name} built ({target.stat().st_size / GIB:.2f} GiB)")
    print(f"\n  The f16 intermediate ({f16.stat().st_size / GIB:.1f} GiB) can be deleted "
          "unless you plan to build another quantisation.")
    return True


def _ensure_converter() -> bool:
    """Fetch llama.cpp's conversion source and the packages it needs."""
    if (CONVERT_SRC / "convert_hf_to_gguf.py").is_file():
        return True
    print("  fetching conversion tools ...", flush=True)
    code, output = _run([sys.executable, "-m", "pip", "install", "--quiet",
                         *BUILD_PACKAGES], timeout=1800)
    if code != 0:
        _fail(f"build packages: {output.splitlines()[-1][:160] if output else code}")
        return False

    url = f"https://github.com/ggml-org/llama.cpp/archive/refs/tags/{LLAMA_RELEASE}.zip"
    archive = ROOT / "tools" / "_llamacpp_src.zip"
    try:
        urllib.request.urlretrieve(url, archive)
        with zipfile.ZipFile(archive) as zf:
            root_name = zf.namelist()[0].split("/")[0]
            wanted = (f"{root_name}/convert_hf_to_gguf.py",
                      f"{root_name}/conversion/", f"{root_name}/gguf-py/gguf/")
            for member in zf.namelist():
                if member.startswith(wanted) and not member.endswith("/"):
                    destination = CONVERT_SRC / member[len(root_name) + 1:]
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(zf.read(member))
    except Exception as exc:  # noqa: BLE001
        _fail(f"could not fetch conversion tools: {exc}")
        return False
    finally:
        archive.unlink(missing_ok=True)
    return (CONVERT_SRC / "convert_hf_to_gguf.py").is_file()


def configure_thin_client(host: str) -> bool:
    """Point this machine at a remote AI server and database."""
    _step("Thin client configuration")
    host = host.strip().rstrip("/")
    if "://" in host:
        host = host.split("://", 1)[1]
    host = host.split(":")[0]

    ai_url = f"http://{host}:8077"
    dsn = f"postgresql+psycopg://saledeed:saledeed@{host}:5432/saledeed"
    print(f"  SALEDEED_AI_URL={ai_url}")
    print(f"  SALEDEED_DB_URL={dsn}")
    print("\n  Set these in your environment, then start the UI with:")
    print("    python -m app.main")
    print("\n  On the host machine, the AI server must listen on all interfaces:")
    print("    python -m ai_server.server --host 0.0.0.0")
    print("  and PostgreSQL must accept remote connections "
          "(listen_addresses and pg_hba.conf).")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------



def install_translation_model(repo: str | None = None,
                              force: bool = False) -> bool:
    """Download the multilingual translation model if it is not already here.

    Idempotent by design: a present, complete model is left alone. The check is
    for *weights*, not for the directory - a partial download leaves the config
    and tokenizer behind, and treating that as "installed" produces a system
    that reports itself ready and fails on the first document.

    The model is fetched into the OCR virtual environment's interpreter, because
    that is where `huggingface_hub` lives and where the model will later run.
    """
    from core.translation import DEFAULT_MODEL, DEFAULT_MODEL_REPO, MODEL_ROOT

    repo = repo or DEFAULT_MODEL_REPO
    target = MODEL_ROOT / (repo.split("/")[-1] or DEFAULT_MODEL)

    def weights_present() -> bool:
        return bool(list(target.glob("*.safetensors"))
                    or list(target.glob("pytorch_model*.bin")))

    if weights_present() and not force:
        size = sum(f.stat().st_size for f in target.rglob("*") if f.is_file())
        _ok(f"translation model already present ({size / 1024 ** 3:.1f} GB) "
            f"- skipping download")
        return True

    python = _ocr_python()
    if python is None:
        _fail("no interpreter with huggingface_hub - run --install-ocr first")
        return False

    print(f"  downloading {repo} (~2.5 GB, once)")
    target.mkdir(parents=True, exist_ok=True)
    code = (
        "from huggingface_hub import snapshot_download;"
        f"snapshot_download({repo!r}, local_dir={str(target)!r},"
        " allow_patterns=['*.json','*.model','*.safetensors','*.bin','*.txt'])"
    )
    result = subprocess.run([str(python), "-c", code], text=True,
                            encoding="utf-8", errors="replace", check=False)
    if result.returncode != 0:
        _fail(f"download failed (exit {result.returncode})")
        return False

    if not weights_present():
        _fail("download completed but no weights are present - "
              "the repository may publish them under another name")
        return False

    return verify_translation_model(target)


def verify_translation_model(target: "Path | None" = None) -> bool:
    """Check the model is complete enough to load.

    A truncated download is the failure worth catching here: the files exist,
    the directory looks right, and the model raises somewhere deep in
    `from_pretrained` on the first document of a 500-document batch.
    """
    from core.translation import DEFAULT_MODEL, MODEL_ROOT

    target = target or (MODEL_ROOT / DEFAULT_MODEL)
    if not target.is_dir():
        _fail(f"model directory missing: {target}")
        return False

    required = ["config.json", "tokenizer_config.json"]
    missing = [name for name in required if not (target / name).is_file()]
    if missing:
        _fail(f"model is incomplete - missing {', '.join(missing)}")
        return False

    weights = list(target.glob("*.safetensors")) + list(target.glob("pytorch_model*.bin"))
    if not weights:
        _fail("no model weights found")
        return False

    total = sum(f.stat().st_size for f in weights)
    if total < 512 * 1024 * 1024:
        _fail(f"weights are only {total / 1024 ** 2:.0f} MB - truncated download")
        return False

    # Confirm the tokenizer actually declares the languages we depend on. A
    # model that loads but lacks Kannada would fail silently at the first deed.
    try:
        import json

        config = json.loads((target / "config.json").read_text(encoding="utf-8"))
        _ok(f"translation model verified: {target.name} "
            f"({total / 1024 ** 3:.1f} GB, {config.get('model_type', '?')})")
    except Exception:  # noqa: BLE001
        _ok(f"translation model verified: {target.name} "
            f"({total / 1024 ** 3:.1f} GB)")
    return True


def _ocr_python() -> "Path | None":
    """The interpreter that hosts Surya and the translator."""
    for relative in ("models/SuryaOCR/venv_new/Scripts/python.exe",
                     "models/SuryaOCR/venv/Scripts/python.exe",
                     "models/SuryaOCR/venv_new/bin/python",
                     "models/SuryaOCR/venv/bin/python"):
        candidate = paths.ROOT / relative
        if candidate.is_file():
            return candidate
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true",
                        help="install everything this machine's class needs")
    parser.add_argument("--install-deps", action="store_true")
    parser.add_argument("--install-runtime", action="store_true")
    parser.add_argument("--install-database", action="store_true")
    parser.add_argument("--build-model", action="store_true")
    parser.add_argument("--install-translation", action="store_true",
                        help="download the multilingual translation model")
    parser.add_argument("--verify-translation", action="store_true",
                        help="check the translation model without downloading")
    parser.add_argument("--force-download", action="store_true",
                        help="re-download even if the model is present")
    parser.add_argument("--quant", default=None,
                        help="override the quantisation chosen for this class")
    parser.add_argument("--thin-client", metavar="HOST",
                        help="configure as a client of a remote host")
    parser.add_argument("--db-password", default="saledeed")
    args = parser.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except AttributeError:
        pass

    readiness = assess(ROOT)
    print(readiness.report())
    plan = readiness.plan

    if args.thin_client:
        return 0 if configure_thin_client(args.thin_client) else 1

    steps: list[tuple[str, object]] = []
    if args.all:
        steps.append(("deps", lambda: install_dependencies(plan.needs_model)))
        if plan.deployment is DeploymentClass.THIN_CLIENT:
            print("\n  This machine is a thin client. Run with --thin-client HOST "
                  "to point it at the host, then start the UI.")
            return 0
        if plan.needs_cuda or plan.deployment is DeploymentClass.CPU_ONLY:
            steps.append(("runtime", lambda: install_runtime(
                cpu_only=plan.deployment is DeploymentClass.CPU_ONLY)))
        steps.append(("database", lambda: install_database(args.db_password)))
        # Translation is part of a complete install: the export is specified to
        # be English, and without this the pipeline writes the source language.
        steps.append(("translation", lambda: install_translation_model(
            force=args.force_download)))
        if plan.needs_model:
            quant = args.quant or plan.quantisation
            steps.append(("model", lambda q=quant: build_model(q)))
    else:
        if args.install_deps:
            steps.append(("deps", lambda: install_dependencies(True)))
        if args.install_runtime:
            steps.append(("runtime", lambda: install_runtime(
                cpu_only=plan.deployment is DeploymentClass.CPU_ONLY)))
        if args.install_translation:
            steps.append(("translation", lambda: install_translation_model(
                force=args.force_download)))
        if args.verify_translation:
            steps.append(("translation", verify_translation_model))
        if args.install_database:
            steps.append(("database", lambda: install_database(args.db_password)))
        if args.build_model:
            quant = args.quant or plan.quantisation or "Q4_K_M"
            steps.append(("model", lambda q=quant: build_model(q)))

    if not steps:
        print("\nNo action requested. Use --all to install what this class needs.")
        return 0 if readiness.ready else 1

    for name, action in steps:
        if not action():  # type: ignore[operator]
            print(f"\nStopped at '{name}'. Fix the problem above and re-run - "
                  "completed steps are skipped.")
            return 1

    print("\n--- re-assessing ---")
    final = assess(ROOT)
    print(final.report())
    if final.ready:
        print("\nSetup complete. Start with:")
        print("    python -m ai_server.server        (terminal 1)")
        print("    python -m app.main                (terminal 2)")
        return 0
    print(f"\n{len(final.missing)} prerequisite(s) still missing.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

# `ai_server/` — inference

A separate process on purpose. It links CUDA and holds ~3.2 GiB of weights; the
desktop shell must be able to start, run and exit without either.

| Module | Does |
|---|---|
| `server.py` | the HTTP endpoints — stdlib `ThreadingHTTPServer`, **not** FastAPI |
| `profiles.py` | picks quantisation, context length and KV cache from the hardware |
| `hardware.py` | CPU, RAM, GPU, VRAM, disk |
| `resources.py` | `ResourceGovernor` — exclusive GPU access below 12 GiB, so OCR and the LLM never co-reside |
| `deployment.py` | what this machine can actually run, and what it must refuse |
| `engines/` | `base.py` the interface · `llamacpp.py` production · `mock.py` a deterministic stub for tests |

## Two constraints that are not negotiable

**The extraction model is fine-tuned and specific to this project.** Verify it,
never download it. An installer that helpfully fetched "a Gemma 3 4B" would
replace the weights every accuracy figure in this project was measured against,
and nothing downstream would detect the substitution.

**Only the dedicated NVIDIA GPU.** The integrated AMD adapter is excluded by
design, not by oversight.

## One trap worth naming

`models/SuryaOCR/server.py` imports FastAPI. It is the vendor's own standalone
Surya server, nothing in this project calls it, and FastAPI is not installed. An
audit of the tree will find that import and conclude FastAPI is part of the
stack. It is not — leave the file alone and do not install its dependencies.

# Project Memory — saledeed v3

Key facts and gotchas to remember across sessions.

## Model: Gemma-3-4B deeds v6.7 (vision)
- Location: `models/AI server/gemma4b/` (merged model, model.safetensors ~8.6 GB bf16).
- Architecture: `Gemma3ForConditionalGeneration`, multimodal (SigLIP vision tower, 896x896, 256 image tokens/image). **Vision works** — can extract the deed schema directly from a page image, no OCR needed.
- Source: Cloudflare R2 bucket `serverai`, key prefix `saledeed/gemma-3-4b-it-deeds-v6_7-merged/`. Creds in `models/AI server/bucket_cred.txt` (plaintext — keep out of any shared/VC location).

## Environment gotchas (must match, else broken)
- **transformers must be 5.5.0** to load this checkpoint. It uses the old key layout `model.language_model.model.layers.*` with tied embeddings (no separate lm_head). transformers 5.8.x expects `model.language_model.layers.*` and silently RANDOM-INITIALIZES the language model (load runs, output is garbage). config.json records the save version = 5.5.0.
- **GPU = RTX 5060 Ti 16GB, Blackwell sm_120.** Needs PyTorch CUDA 12.8+ wheels: `pip install torch --index-url https://download.pytorch.org/whl/cu128`. The default/CPU torch (or cu124) will not run on it. Driver 591.86 / CUDA 13.1.
- Model load ~4s, generation ~14s for ~320 tokens on the 5060 Ti. CPU inference is effectively unusable (too slow) — always use the GPU.
- The R2 download was **missing `preprocessor_config.json`**; it was recreated with standard Gemma3-4B vision values. If re-downloading the model, that file must be present for the vision processor to load.

## Prompt / task
- Finetuning prompt: `models/saledeed main/prompt_v6_short.txt` (also copied to `tests/corpus/test scripts/prompt.txt`). Outputs one JSON object: buyer_details[], seller_details[], property_details{}, document_details{}.

## Input mode: USE OCR TEXT, NOT IMAGES
- The model was **finetuned on OCR text**, so text is its native input; images are out-of-distribution.
- Tested both on 140-2024-25.pdf: vision (whole-doc images) duplicated the buyer into sellers, truncated Aadhaar digits, garbled addresses, and put stamp duty in registration_fee. **OCR-text mode fixed all of these** (correct 12-digit Aadhaars, one buyer/one seller, clean addresses, registration_fee 33000, consideration 3300000). Feed OCR text.
- OCR source used = the PDF's **embedded text layer** (fitz `page.get_text()`); these registration PDFs ship an OCR layer. English fields are clean; Kannada boilerplate is garbled but the key fields here are English.
- **Surya (surya-ocr 0.17.1) is INCOMPATIBLE with transformers 5.5.0** (KeyError 'default' in ROPE_INIT_FUNCTIONS + new rope_parameters config API). To use Surya for higher-quality OCR, run it in a SEPARATE env/process from Gemma. Tesseract is installed but requires admin elevation (can't run here).

## Workload shape & serving engine (for the app)
- **Stateless single-turn**: one deed -> one JSON. No multi-turn, no conversation history.
- **No KV reuse across requests**: every deed's OCR text is unique. Only the ~300-token instruction/schema prefix is shared (negligible vs 7k-22k-token OCR bodies), so prefix caching barely helps. Throughput gain must come from CONTINUOUS BATCHING, not caching.
- **Prefill-heavy**: long unique prompts, short outputs (~300-850 tok). Tune `--max-num-batched-tokens` (chunked prefill) + `--max-num-seqs`; expect a real but modest throughput gain (prefill-bound).
- **Engine chosen: vLLM via SystemPanic/vllm-windows fork v0.20.0** (prebuilt wheel; Py3.12+CUDA13+torch2.11+Blackwell match this box). vLLM = continuous batching + PagedAttention + OpenAI server. Alternatives: ExLlamaV3+TabbyAPI (needs exl3 quant); TGI deprecated; llama.cpp/Ollama weaker for production batching.
- Caveats to resolve at install: wheel wants torch 2.11+cu130 (we have cu128); checkpoint old-layout (language_model.model.layers) may need a load-test/re-save under vLLM's loader.
- HF batched `generate` measured only ~16% faster at batch=3 (peak 16.1GB VRAM, batch=4 OOMs) because it waits for the longest sequence — this is why we move to vLLM.

## vLLM environment (isolated)
- **Location: `C:\vl`** (a venv, NOT in the repo). Kept short on purpose — vLLM's FlashInfer cubin filenames are 160+ chars and blow past the Windows 260-char path limit under the long repo path `d:\saledeed v3\...`.
- **Windows Long Path support was enabled** (HKLM\SYSTEM\CurrentControlSet\Control\FileSystem\LongPathsEnabled=1, set by user via admin) — required for the FlashInfer cubins to extract. Without it the install rolls back.
- Wheel: `vllm-0.23.0+cu132-cp312-cp312-win_amd64.whl` from github.com/SystemPanic/vllm-windows (still in repo root as the installer; can delete). Installed with `--extra-index-url https://download.pytorch.org/whl/cu130` -> torch 2.11+cu130 in C:\vl.
- This env is SEPARATE from the global Python (which runs the validated HF pipeline: torch 2.11+cu128 + transformers 5.5.0). Use `C:\vl\Scripts\python.exe` for vLLM, the global python for HF.
- Benchmark: `tests/corpus/test scripts/bench_vllm_batching.py` (offline + server modes).

### vLLM bring-up status (as of 2026-06-22): model loads & runs, BLOCKED on driver
PROVEN: vLLM CAN run our finetuned Gemma3 checkpoint — it loads, maps weights, initializes, and reaches generation. Required fixes/patches (all done):
1. **vLLM gemma3 weight-mapper patch** (`C:\vl\Lib\site-packages\vllm\model_executor\models\gemma3_mm.py`, .orig backup saved): added `"model.language_model.model.": "language_model.model."` as the FIRST orig_to_new_prefix rule so our OLD-layout weights map correctly. WeightsMapper applies prefix rules in order; after this fires the key no longer matches the next rule. (vLLM env has transformers 5.12 but uses its OWN loader, so the old layout needs this patch.)
2. **flashinfer cudart patch** (`C:\vl\Lib\site-packages\flashinfer\jit\__init__.py`, .orig saved): robust best-effort cudart preload (globs cuda_path/bin/** and torch/lib; never hard-fails).
3. **Copied CUDA runtime DLLs** from `nvidia\cu13\bin\x86_64\` UP to `nvidia\cu13\bin\` (cudart64_13.dll, nvrtc64_130_0.dll, nvrtc-builtins64_133.dll, nvvm64_40_0.dll) — flashinfer looks in bin/ and bin/x64, not bin/x86_64.
4. **Required env vars** to launch vLLM here: `CUDA_HOME=C:\vl\Lib\site-packages\nvidia\cu13`, `VLLM_ENABLE_V1_MULTIPROCESSING=0` (avoids zmq port + spawns), `VLLM_USE_FLASHINFER_SAMPLER=0` (avoids flashinfer JIT for sampling). For text-only also pass `limit_mm_per_prompt={"image":0}`.

RESOLVED (2026-06-22): driver updated 591.86 -> **610.62** (CUDA 13.2+). FLASH_ATTN cu132 kernels now load. **vLLM WORKS** — generates correct output (`{"ok": true}` load test passed).

### vLLM WORKING — launch recipe & benchmark
- Launch env (all required): `CUDA_HOME=C:\vl\Lib\site-packages\nvidia\cu13`, `VLLM_ENABLE_V1_MULTIPROCESSING=0`, `VLLM_USE_FLASHINFER_SAMPLER=0` (MSVC cl.exe still missing, so avoid flashinfer JIT for sampling). LLM(dtype=bfloat16, limit_mm_per_prompt={"image":0}) for text-only.
- PyMuPDF installed in C:\vl venv too (bench reads PDFs there).
- **Benchmark (offline, continuous batching): 32 deed extractions in 56.0s = 0.57 req/s, 7132 input tok/s prefill, 270 output tok/s (~1.75s/deed).** vs HF sequential ~33s/deed -> ~18x throughput. Confirms continuous batching is the right engine for the unique-prompt, prefill-heavy deed workload.
- First-run startup is slow (one-time Triton/CUDA-graph JIT, ~1min); steady-state is fast.
- Default FLASH_ATTN backend works now (no need to force FLASHINFER). Vision tower also unblocked by the driver update but we run text-only.
- Scripts: `bench_vllm_batching.py` (offline+server), `vllm_loadtest.py`, `bench_vllm_ocr.py` (real OCR corpus).

### OCR corpus auto-batch benchmark (2026-06-22)
- Input: 50 real OCR txt files in `tests/corpus/OCR saledeeds/` + `models/saledeed main/prompt_v6_short.txt`. Token sizes: min 5.4k, median 9.4k, max 33k.
- Script `bench_vllm_ocr.py`: auto-sizes max_model_len (= longest_prompt+max_new, rounded), saves each JSON to `outputs/vllm_ocr/`.
- **gpu_memory_utilization: 0.95 FAILS** — the 5060 Ti is the DISPLAY GPU (~1.1GB/7% used by Windows desktop), only ~14.82/15.93 GiB free at startup. Use **0.90** (=10% headroom) which fits. 0.95 needs a headless/second GPU.
- **Result: 50 deeds in 318.7s, gpu 99% util. 536k prefill tok (1682 tok/s), 33k output tok (104 tok/s), avg 664 output tok/deed.** 49 deeds finished in ~178s; 1 straggler took ~138s.
- **Quality: 48/50 valid JSON (96%); good field extraction (parties, PANs, Aadhaars, addresses).** 2 failures = greedy-decode REPETITION LOOPS (2732 looped "Shivalingappa Layout,..." to the 8000 cap = the 138s straggler; 2725 appended a duplicate JSON).
- **TUNING: max_new=8000 lets runaway loops dominate wall time. Cap max_new ~1536-2048 (legit outputs avg 664, max ~1300) and/or add repetition_penalty ~1.1 to kill loops — would cut the tail straggler from ~138s to ~30s and tighten throughput.**



## Test harness
- `tests/corpus/test scripts/test_gemma_vision_deed.py` — renders `tests/corpus/saledeeds/*.pdf` pages and runs the model on GPU. Output -> `tests/corpus/test scripts/outputs/`.

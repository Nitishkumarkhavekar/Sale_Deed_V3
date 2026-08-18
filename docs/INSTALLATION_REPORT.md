# Installation Report

**Generated:** 2026-08-18T13:18:28
**Duration:** 85s

## System

| Property | Value |
|---|---|
| windows | Windows 11 (10.0.26200) |
| architecture | AMD64 |
| 64bit | True |
| hostname | DESKTOP-862KV07 |
| python_running | 3.13.14 |
| administrator | False |
| cpu | AMD64 Family 23 Model 96 Stepping 1, AuthenticAMD |
| cores_physical | 8 |
| cores_logical | 16 |
| ram_total_gb | 7.4 |
| ram_available_gb | 2.2 |
| cuda_available | True |
| cuda_version | 12.7 |
| driver_version | 566.07 |
| gpu | NVIDIA GeForce RTX 3050 Laptop GPU (4.0 GiB) |
| gpu_vram_gb | 4.0 |
| disk_free_gb | 127.2 |
| disk_total_gb | 195.3 |
| internet | True |
| git | git version 2.52.0.windows.1 |
| postgresql | service installed, not reachable |

## Steps

| Step | Result | Detail | Seconds |
|---|---|---|---|
| Folders | installed | 1 created | 0.0 |
| Visual C++ runtime | found | present | 0.0 |
| Python 3.12 (OCR) | found | present | 0.1 |
| Environment | found | project virtualenv .venv | 0.0 |
| Python packages | installed | 6 installed | 70.7 |
| Migrations | found | already at head | 1.6 |
| Configuration | found | .env exists - left untouched | 0.0 |
| Inference runtime | found | llama-server.exe | 0.0 |
| Extraction model | found | deeds-v6_7-f16.gguf (7.24 GB) | 0.0 |
| vLLM engine | found | ready in models/vllm-env | 8.2 |
| Translation model | found | nllb-200-distilled-600M (2.3 GB) | 0.0 |
| OCR (Surya) | found | via venv_new | 0.2 |

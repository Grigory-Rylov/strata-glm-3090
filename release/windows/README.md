# Strata NVFP4 — ready-made engine for Windows

Qwen3.8-Flash-Next (125B hybrid MoE), OrcaRouter's abliteration in ModelOpt NVFP4, on one GeForce RTX 50-series card.
Source, measurements and how it works: https://github.com/sergqwer/strata-nvfp4

## What you need

- **GPU:** GeForce RTX 50-series (Blackwell, sm_120). Built and measured on an RTX 5090 (32 GB); a card with less
  VRAM runs with a smaller expert cache and decodes slower. Other GPU generations are not supported by this build.
- **Driver:** NVIDIA 580 or newer (CUDA 13). The CUDA runtime is inside `strata.exe` and cuBLAS is in `engine\`,
  so no CUDA toolkit is needed.
- **RAM:** 96 GB or more (the experts take 63 GiB, pinned, while the model runs); measured with 128 GB.
- **CPU:** any x86-64 with AVX2; AVX-512 (Zen 4/5) is used automatically for the CPU share of the experts.
- **Disk:** ~300 GB free while the model is prepared, ~170 GB afterwards. The start reads 63 GiB, so the fastest
  NVMe drive you have is the right place for this folder (~8 s to start from PCIe 5, ~15 s from PCIe 4).
- **Python 3.11 or newer** on PATH (the scripts make their own virtual environments here).

## Two steps

1. **`prepare-model.cmd`** — downloads
   [jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4)
   (126 GB) and converts it into `models\`, the n-gram (PLE) table kept in FP8 and the token embedding in BF16
   exactly as Qwen ships them. It takes a while; if it stops, run it again and it resumes. Coming from an older
   release: run it again too - it adds only `models\token-embd-bf16.gguf` (1.3 GB; the checkpoint must still be in
   `models\checkpoint`).
2. **`start-server.cmd`** — loads the model and serves it on **http://127.0.0.1:8080**:
   - a chat page at `http://127.0.0.1:8080/`
   - OpenAI API at `/v1/chat/completions`, Anthropic API at `/v1/messages` (Claude Code can point at it)

First start after a reboot is slower while Windows reads the files; later starts take ~8-15 s.

## Measured (RTX 5090, Ryzen 9 9950X3D, 128 GB DDR5-5600, 262K context)

| | |
| --- | ---: |
| Writes answers, short chat | ~115 tokens/s |
| Writes answers, 32K context | ~120 tokens/s |
| Reads a 32K prompt | ~5,000 tokens/s |

## Settings

`config\strata-nvfp4.json` holds the engine flags: context (`--max-context`, up to 262144), KV cache (`--kv int8`),
paths. Environment switches:

- `STRATA_PREFILL_NVFP4=w4a8` (default) / `w4a4` (faster prompt reading, less accurate) / `fp16` (exact, slower)
- `--pcie-frac F` in the config: the share of cache misses fetched over PCIe (default 0.25)

## Large pages (optional, needs one sign-out)

Not more memory — bigger pages. The 63 GiB of experts are 16.5 million 4 KB pages; with 2 MB pages they are 32
thousand, and the CPU share of every token runs steadier (7.3-7.7 ms per step instead of 7.4-11 on the reference
PC). Windows allows it only for an account with *Lock pages in memory*, and a new right reaches you only at the
next sign-in:

1. run **`enable-large-pages.cmd`** (it asks for admin; works on Windows Home too),
2. **sign out and back in, or reboot** — locking the screen is not enough,
3. `enable-large-pages.cmd -Check` should now say ACTIVE, and `strata.log` shows `large pages (2097152 B)`.

`large pages refused ... error 1314` in the log = the right is not active yet (step 2); `error 1450` = memory is
too fragmented after a long uptime (reboot). Either way the model still runs, on 4 KB pages.
`enable-large-pages.cmd -Revoke` undoes it.

## Claude Code

```bat
set ANTHROPIC_BASE_URL=http://127.0.0.1:8080
set ANTHROPIC_API_KEY=local
set ANTHROPIC_MODEL=strata-nvfp4
set ANTHROPIC_SMALL_FAST_MODEL=strata-nvfp4
claude
```

Each turn re-reads only what is new: measured, a 21,964-token first request (5.8 s), then 209 and 106 new tokens
per tool turn (~0.5 s).

## Folder

| | |
| --- | --- |
| `engine\` | `strata.exe`, NVIDIA cuBLAS (`cublas64_13.dll`, `cublasLt64_13.dll`), `BUILD.json` |
| `serve\` | the server and its chat page |
| `tools\`, `third_party\llama.cpp\` | the converters `prepare-model.cmd` runs (llama.cpp's at its pinned commit) |
| `data\` | the expert-cache profile and the draft vocabulary |
| `docs\NVFP4.md` | accuracy and speed measurements |

The model is an abliterated fine-tune: it does not refuse. What it is used for is on whoever runs it.

License: MIT (see `LICENSE`); third-party parts in `THIRD-PARTY-NOTICES.txt`.

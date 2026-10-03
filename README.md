# strata-glm-3090 — the archived strata-glm experiment, ported to Linux + RTX 3090

An active fork of [sergqwer/strata-glm](https://github.com/sergqwer/strata-glm) (an archived
experiment that ran **GLM-5.3-Flash** — a 204 GB MoE model, 42 MoE layers × 288 experts,
12,096 routed experts of 14.16 MB each — on one RTX 5090 + 128 GB RAM + two NVMe drives).
Upstream is Windows-only and read-only; this fork runs the same engine on **Linux, CUDA 13.2,
one RTX 3090 (24 GB)** and adds what the port required. The engine itself — the tiered expert
cache (VRAM → RAM → NVMe), the MMQ int8 tensor-core prefill path, the KDA/DSA attention and the
pack format — is upstream's; everything measured below was re-measured here.

## What this fork adds

| area | change |
|---|---|
| Linux port | portable 64-bit seek (`fseeko`); the disk tier (`DiskReader`) via `O_DIRECT` pread per drive thread with a buffered fallback; the RAM tier (`TieredExpertSource`) rebuilt on pread windows + `cudaHostRegister` (upstream had it under `#ifdef _WIN32`) |
| multi-GPU | `--layer-split K1,K2,.. --gpus D0,D1,..`: layer-pipeline stages, each stage with its own VRAM expert cache (bit-exact against the single-stage engine) |
| pack builder | `tools/glm_pack.py` runs on Linux (`/proc` private-memory guard instead of `psapi`, language-model tensor-prefix resolution for the checkpoint's `model.language_model.*` names) |
| docs | `PROBLEMS.md` — the measured journal of every port step; the 3090 numbers below |

## Results (1× RTX 3090 24 GB, 251 GB RAM, GPU2, CUDA 13.2, test pack)

Measured 2026-10-03 on a synthetic pack (real GLM-5.3-Flash geometry, random values — the byte
traffic and kernels are the real ones, the text quality is not; real-weight numbers follow the
pack build from the canonical checkpoint):

| | prefill | decode |
|---|---|---|
| disk-only tier (the port's first milestone) | — | 1.15 tok/s |
| **RAM tier, `--chunk 4096`, 16K-token prompt** | **437.8 tok/s** | **19.8 tok/s** |
| RAM tier, `--chunk 4096`, 64K-token prompt | 420.7 tok/s | — |
| RAM tier, short 1K prompt, `--chunk 2048` | 132 tok/s | 19.8 tok/s |

Two regimes, one wall: a short prompt streams nearly all 12,096 experts (171 GB) over PCIe for
~1K tokens — the bus caps it near 190 tok/s; a long prompt amortizes the same stream 16–64× and
the MMQ compute path shows what the engine is for. `--chunk 8192` (upstream's 32 GB recipe)
OOMs on 24 GB in `init_prompt_mmq` — use 4096.

Decode split (49.8 ms/token): expert copies 12.4 + expert kernels 0.2 + the rest (dense layers,
routers, syncs) 37.2 — dense/sync is the current ceiling, not the expert stream.

## Build (Linux)

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF
make -C build strata-glm -j16
```

CMake fetches the pinned ggml commit; on a busy network add
`-DFETCHCONTENT_SOURCE_DIR_STRATA_LLAMACPP=<llama.cpp checkout>` to take it locally.

## Run

```
CUDA_VISIBLE_DEVICES=2 GLM_TIMING=1 ./build/strata-glm \
  --pack <pack> --tokens <token_ids.txt> --chunk 4096 \
  --max-context <prompt+new> --max-new 256 --skip-disk 0 --skip-ram 0 \
  --dense-fp4 all --profile <profile.bin>
```

The full step-by-step (pack layout, `native_experts.txt`, `profile.bin` format, expected log,
limits) lives in the workstation runbook: `408032gb/docs/strata-tiered-run.md`.

## Weights

Canonical GLM-5.3-Flash weights for this deployment:
`/mnt/data/home/grishberg/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-*-of-00006.gguf`
(GGUF; the engine reads its own pack format, a GGUF→pack converter is the pending piece).
`tools/glm_pack.py` packs the ModelOpt NVFP4 checkpoint layout (`weight` u8 nibbles +
`weight_scale` e4m3 + `weight_scale_2` f32 per projection) losslessly into the engine's
block format; its upstream download instructions are legacy here.

## Upstream

Built on the [Strata](https://github.com/Niko1221/Strata) engine via the
[strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) fork (its README is kept as
[README.strata-nvfp4.md](README.strata-nvfp4.md)); the prompt path's MMQ kernels come from
llama.cpp's ggml. Upstream's quality table (dense quantizations vs a BF16-dense reference, KL
per step) and the archived project narrative remain in
[docs/DETAILS.md](docs/DETAILS.md) and the git history. See `PROBLEMS.md` for this fork's
measured journal: the Linux port, the 3090 speed ladder, and the prefill regime math.

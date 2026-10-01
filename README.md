# strata-glm — GLM-5.3-Flash NVFP4 on one desktop (archived experiment)

A native C++/CUDA engine that runs **GLM-5.3-Flash in NVFP4** — a 204 GB MoE checkpoint (42 MoE layers × 288
experts, 12,096 routed experts of 14.16 MB each) — on **one RTX 5090 (32 GB) + 128 GB RAM + two NVMe drives**,
without GGUF and without a GPU cluster. It is built on the [Strata](https://github.com/Niko1221/Strata) engine
(via the [strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) fork, whose README is kept as
[README.strata-nvfp4.md](README.strata-nvfp4.md)); everything GLM-specific lives in `src/glm/` and `tools/glm_*`.

**Status: archived.** It works and is ~6× faster than llama.cpp on the same box, but ~15–18 tok/s of decode
was not usable for the author's workload, and the hardware leaves little headroom (see *Where the limit is*).
There is no server / chat integration — the engine takes token ids and prints token ids.

## Results (RTX 5090, 128 GB DDR5, Samsung 9100 PRO + 990 PRO, Windows 11, CUDA 13.3)

| | prefill | decode |
| --- | ---: | ---: |
| llama.cpp, UD-IQ4_XS GGUF, both SSDs (the starting point) | 29 tok/s | 2.4–2.9 tok/s |
| strata-glm, chat prompt (~1–1.4K tokens), 192 decode steps | — | **18.0** (Cyrillic chat) / **14.8** (code) |
| strata-glm, same, exact mode (`--skip-disk 0`) | — | 16.5 / 12.9 |
| strata-glm, 64K-token prompt at `--max-context 262144` | **~1000 tok/s** | 15.3 (exact mode) |
| strata-glm, opt-in `--skip-disk 0.15 --skip-ram 0.05` | — | 20.3 / 16.4 |

Decode went 1 → 5.2 → 7.9 → 9.8 → 12–14.5 → 16.5 → 18 tok/s over the project (file reads → LRU tiers → mirrored
drives + prefetch → FP8/int8 dense → static VRAM tier → lazy context caches → skipping tiny disk experts).
Prefill went 62 → 156 (MMQ) → 478 → 857 (KDA scan in registers) → ~1000 tok/s (streamed expert pool).

### Accuracy

KL divergence to a run with BF16 dense weights and an FP16 MLA latent (the routed experts are the checkpoint's
NVFP4 in every case), teacher-forced; decode is deterministic, so the noise floor is 0. "Long" = 48 steps after
a 2600-token prompt, "short" = 129 steps after an 849-token code prompt.

| format / option | long: KL mean / median, top-1 | short: KL mean / median |
| --- | --- | --- |
| FP8 E4M3 dense, a scale per row (the early default) | 0.018 / 0.0135, 93.8% | 0.023 / 0.0021 |
| int8 dense, an FP16 scale per 32 values | 0.011 / 0.0078, 91.7% | 0.019 / 0.0007 |
| **default**: int8 dense + NVFP4 KDA q/k + int8 latent | 0.018 / 0.017, 91.7% | 0.018 / 0.0024 |
| default + `--skip-disk 0.1` (now the default) | 0.021 / 0.016, 93.8% | 0.017 / 0.0010 |
| `--dense-fp4 all` (every dense matrix NVFP4) | 0.048 / 0.041, 83.3% | 0.038 / 0.0039 |
| 3-bit experts (fake-quantized, rejected) | 0.19 / 0.14, 79.2% | — |

Lesson: a short prompt hides quantization error — all-NVFP4 dense looked fine on the short test and was 5× worse
than int8 after 2600 tokens.

## How it works

- **Three tiers for the experts.** VRAM (~1550 slots): the hottest 70% by the prompt's routing are *static* —
  never evicted and **not duplicated in RAM**, so RAM caches ~1100 more experts and the disk is read 30–40% less
  — and re-chosen every 8 decode tokens; the other 30% is an LRU over RAM. Pinned RAM (~7070 experts, leaving
  24 GiB free for Windows) evicts by a decayed estimate of use. The rest is read unbuffered from `experts.bin`
  on two drives holding identical copies, each 1.4 MB piece going to the drive that will get to it first.
- **Prefetch.** At layer *l* a cheap predictor runs layer *l+1*'s router on the current streams (top-1/2/3
  accuracy 95/86/77%): the two most confident RAM experts are copied to VRAM on a separate stream, up to six
  disk-only ones are read into RAM at low priority.
- **Prompt path.** MMQ (int8 tensor cores, from llama.cpp's ggml) over a streamed pool that loads all 288 experts
  of a layer while its attention computes; KDA scan with its state in registers; MLA and the DSA indexer on
  tensor cores (FP16 WMMA); DSA queries in sub-chunks of 2048 so 262K fits.
- **Formats.** Dense weights int8 with a scale per 32 values (NVFP4 for KDA's q/k projections), the MLA latent
  cache int8, the context caches grown with the context instead of allocated for `--max-context` up front.
- **`--skip-disk 0.1`** (default): a routed expert that is only on disk, was not predicted, and carries under 10%
  of the routing weight is left out (the other weights renormalized) instead of waited for.

Offline cache simulators replayed on logged routing (`tools/glm_policy/`, `tools/glm_cache_sim.py`) chose the
policies before they were built; the engine matched them (e.g. disk reads per token 30.6 → 22.2, predicted 22.6).

## Where the limit is

Per token the model routes to 336 experts (4.76 GB). With 13% of the experts in VRAM and 58% in RAM, a chat
token moves ~2.2–3.2 GB over PCIe (measured 50.8 GB/s H2D; the link is ~80% busy) and reads ~10–20 experts from
disk. A profile shows the GPU idle only while a layer waits for a disk expert the predictor missed (half of those
were not even in its top 16; reading deeper predictions costs more disk time than it saves). An offline Belady
bound puts the best possible VRAM hit rate at 73% against 55–58% reached. 100 tok/s is out of reach for this
model on this hardware without dropping experts.

Tried and dropped: fully exclusive RAM/VRAM tiers (write-backs cost more than they save), CPU computing part of
the RAM experts, staging disk reads straight to VRAM, lossless compression of the experts (FP4 code entropy
3.97 of 4 bits, lzma 97.6%), 3-bit experts (KL 0.19), deeper speculative copies / disk prefetch, disk-queue
priority tweaks, split expert kernel launches.

## Building and running (Windows)

Visual Studio Build Tools 2022, CUDA 13.x, CMake, Ninja; sm_120 (Blackwell) was the target.

```bat
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF ^
      -DCMAKE_CUDA_ARCHITECTURES=120a
cmake --build build --target strata-glm
```

Without `-DSTRATA_GGML_DIR=<llama.cpp checkout>` CMake fetches the pinned ggml commit.

1. Download the ModelOpt checkpoint `nvidia/GLM-5.3-Flash-NVFP4`.
2. Pack the experts (lossless repack into ggml's block_nvfp4 rows, ~171 GB):
   `python tools/glm_pack.py --model <checkpoint> --out <pack>`; optionally copy `experts.bin` to a second drive.
3. An expert profile (the boot ranking of the tiers): `tools/glm_profile.py split / run / stats --profile`.
4. Run on a file of token ids (tokenize with the checkpoint's `tokenizer.json`):

```bat
build\strata-glm.exe --pack <pack> --profile <pack>\expert-profile-boot.bin --mirror D:\<copy>\experts.bin ^
    --tokens prompt_ids.txt --chunk 8192 --max-context 262144 --max-new 256
```

Useful options: `--skip-disk W` / `--skip-ram W` (0 = exact), `--dense-fp4 GROUPS|none`, `--dense-fp8`,
`--latent-f16`, `--vram-static F`, `--restatic N`, `--pf-copies N`, `--pf-reads N`, `--ram-reserve-gib G`,
`--vram-reserve-mib M`, `--teacher FILE --dump-step-logits FILE` (teacher-forced accuracy runs),
`--prompt-f32` (bit-reproducible prompt path). Env: `GLM_TIMING=1` (per-token time split), `GLM_NSYS=1|2`
(an nsys capture range), `GLM_FAKE3=1` (3-bit fake quantization test).

`tools/glm_ref.py` is an FP32 reference forward (layer by layer from the safetensors) and `tools/glm_compare.py`
compares the engine against it. The full development log with every measurement (in Ukrainian) is
[docs/glm-port-plan.md](docs/glm-port-plan.md).

## Credits and license

MIT, as Strata. The engine core, tiers and NVFP4 machinery are [Strata](https://github.com/Niko1221/Strata)'s
and the [strata-nvfp4](https://github.com/sergqwer/strata-nvfp4) fork's; the prompt path's MMQ kernels come from
[llama.cpp](https://github.com/ggml-org/llama.cpp)'s ggml; the checkpoint is NVIDIA's ModelOpt NVFP4 export of
GLM-5.3-Flash. The GLM port was written with Claude (Anthropic).

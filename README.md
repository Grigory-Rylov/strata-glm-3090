# Strata NVFP4

**Qwen3.8-Flash-Next (125B hybrid MoE) in NVFP4 on one RTX 5090 + 128 GB of RAM.** A fork of
[Niko1221/Strata](https://github.com/Niko1221/Strata) that runs a ModelOpt **NVFP4** checkpoint — here
[OrcaRouter's abliterated Flash-Next](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4) —
instead of the Q2/Q3 quants Strata ships for. NVFP4 keeps the experts at 4.5 bits with calibrated scales, which is
why this fork exists: the 2-3 bit quants were noticeably less accurate on the same model.

Strata itself keeps the 63 GiB of routed experts in RAM, caches the most-used ones in VRAM, computes the misses on
the CPU and over PCIe in parallel with the GPU, and decodes with an MTP draft head. Everything about that design is
upstream's; the original README is kept as [README.upstream.md](README.upstream.md).

## How it differs from upstream Strata

- **Runs an NVFP4 checkpoint** (ModelOpt, 4.5-bit experts with calibrated scales) instead of Strata's Q2/Q3 quants:
  lossless repack, per-expert FP32 scales kept and applied to each projection's output.
- **Nothing is rounded coarser than the checkpoint where it can be avoided:** the n-gram (PLE) table in its shipped
  FP8 (upstream reads it as IQ4_NL, 8% off per row), the token embedding in its shipped BF16, RoPE angles from a
  float64 table in every kernel (upstream's fast-math angles were ~0.02 rad off at 262K), FP32-exact inputs to the
  prompt path's BF16 projections (router, indexer, gates), int8 KV behind a Hadamard rotation.
- **An int8 (W4A8) prompt path for Blackwell:** llama.cpp builds NVFP4 MMQ as FP4 x FP4 on sm_120; this keeps
  8-bit activations - 16x smaller error per product - at 86% of its speed.
- **AVX-512 NVFP4 kernels** for the CPU share of the experts (1.8-3.7x ggml-cpu).
- **Faster start:** experts read unbuffered into the pinned arena on their own thread, beside everything else -
  ~8 s to the first token from a PCIe 5 drive.
- **Tuned for NVFP4's larger experts** (PCIe share, prompt chunks up to 32K, fused scale passes, a verify commit
  that overlaps the draft) and the fine-tune's own abliterated MTP draft head.
- **A draft vocabulary with Cyrillic:** the MTP draft head proposes only tokens of its subset, and upstream's held
  142 of the vocabulary's 18,580 Cyrillic tokens - a Ukrainian answer decoded at 83 tokens/s with 1.4 tokens a round;
  with the whole Cyrillic script (`tools/draft_vocab.py --add cyrillic`), 109 and 2.1. English is unchanged.
  Upstream's CJK subset is one `--add cjk` away (`data/draft_vocab_en.bin` is the English/code one).
- **Fixes:** a scale fold that left NVFP4 hidden activations in FP16's subnormals (2-12% expert error), and the
  batched verify path skipping the query rotation of rotated KV caches.
- **Merged with upstream Strata 0.1.28** (its batched draft-layer prompt pass, fused hyper-connection prompt kernels,
  expert grouping through mapped memory, tool-call and cancellation fixes): prompt reading +3-16%, first-token KL
  to the previous build at noise level.

Each change was measured - first-token KL against a reference, and interleaved speed A/B runs;
[docs/NVFP4.md](docs/NVFP4.md) has the numbers, and everything that was tried and dropped.

## Requirements

- **GPU:** RTX 5090, 32 GB (sm_120). The engine fills it: dense weights and the 262K KV cache first, then ~19 GB of
  cached experts (~7,500). A card with less VRAM caches fewer experts and decodes slower.
- **RAM:** 96 GB minimum, measured with 128 GB. A run holds ~69 GiB of physical RAM - 63 GiB of it the pinned
  expert arena.
- **Pagefile:** Windows lets all processes together commit at most RAM + pagefile, and the engine commits ~98 GiB
  (the 69 GiB above plus what WDDM reserves for the GPU's allocations; nothing of the model is ever paged out). With
  Windows and the usual apps on top, RAM + pagefile should be ~140 GB or more: **a pagefile of at least 32 GB with
  128 GB of RAM, 64 GB with 96 GB**. Set a fixed minimum rather than relying on a system-managed file to grow in
  time. Too small, and the start fails with an allocation error.
- **Disk:** ~200 GB for the model files: GGUF 74 GB, expert pack 70 GB, n-gram table 51 GB, embedding 1.3 GB,
  MTP head 0.8 GB. ~340 GB while preparing them (the 135 GB checkpoint and the MTP intermediates can go afterwards).
  Use the fastest NVMe drive you have: every start reads 63 GiB.

## Measured

RTX 5090 (32 GB, PCIe 5 x16), Ryzen 9 9950X3D, 128 GB DDR5-5600, Samsung 9100 PRO, Windows 11, CUDA 13.3.
262,144-token context, KV cache int8, large pages on. The machine is also a desktop: runs vary by ~5%.

| | |
| --- | ---: |
| Writes answers, short chat | ~115 tokens/s (Ukrainian or Russian too: ~110) |
| Writes answers, 32K context | ~120 tokens/s |
| Reads a 32K prompt | ~5,500 tokens/s |
| Start to the first token | ~8 s (63 GiB of experts read at ~10 GiB/s) |

Where precision was still being lost, first-token KL divergence from the more exact variant (8 prompts of 1K-8K
tokens plus one of 32K; two runs of the same configuration differ by a median 0.000007):

| source | KL mean | at 32K | now |
| --- | ---: | ---: | --- |
| RoPE: fast-math float angles | - | 0.0063 | float64 table |
| token embedding stored as Q8_0 | 0.0025 | 0.0041 | BF16 as shipped (`--embd-gguf`) |
| prompt path: BF16-rounded activations into BF16 projections | 0.0023 | 0.0093 | hi + lo split (0.0029 left: not the hyper-connection) |
| n-gram table as IQ4_NL | 0.0026 | - | FP8 as shipped |
| int8 KV without rotation (vs fp16 KV) | 0.0017 | 0.0068 | rotated: 0.0011 / 0.0038 |
| prompt path W4A8 (vs FP16 activations) | 0.0018 | - | default; `fp16` is 24% slower |

## Build (Windows)

Needs Visual Studio 2022 Build Tools, CUDA 13+ (sm_120 wants 13), CMake, Ninja, Python 3.11+. From an x64 Native
Tools prompt:

```bat
git clone https://github.com/ggml-org/llama.cpp third_party\llama.cpp
git -C third_party\llama.cpp checkout 3cf03257f219afbe7334045ff7c6a06ac68c627d
cmake -G Ninja -S . -B build -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120a ^
      -DSTRATA_GGML_DIR=%CD%\third_party\llama.cpp
cmake --build build --target strata
```

Without `-DSTRATA_GGML_DIR` CMake fetches the same llama.cpp commit itself.

## Prepare the model

```bat
python -m venv .venv
.venv\Scripts\python -m pip install numpy torch safetensors transformers sentencepiece
set PYTHONPATH=third_party\llama.cpp\gguf-py

:: 1. the checkpoint (126 GiB)
hf download jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4 --local-dir models\orca-nvfp4

:: 2. the n-gram (PLE) table, 51.2 GB: its FP8 bytes copied as they are (read from the SSD, never loaded into RAM)
.venv\Scripts\python tools\ple_fp8_pack.py --model models\orca-nvfp4 --out models\ple-fp8.gguf

:: 3. the token embedding, 1.3 GB: BF16 as shipped (the GGUF below stores it as Q8_0)
.venv\Scripts\python tools\embd_bf16_pack.py --model models\orca-nvfp4 --out models\token-embd-bf16.gguf

:: 4. GGUF (NVFP4 experts, Q8_0/BF16 dense) and the pack (experts.bin 63 GiB, tokenizer)
.venv\Scripts\python tools\nvfp4_convert.py --model models\orca-nvfp4 --outfile models\orca-nvfp4.gguf
.venv\Scripts\python tools\iq_pack.py --gguf models\orca-nvfp4.gguf --out packs\orca-nvfp4

:: 5. the fine-tune's own MTP draft head
.venv\Scripts\python tools\mtp_extract.py --model models\orca-nvfp4 --out mtp-orca
.venv\Scripts\python tools\mtp_pack.py --src mtp-orca --experts q2_0 --out mtp-orca\mtp-q2_0.gguf
.venv\Scripts\python tools\mtp_rt.py --gguf mtp-orca\mtp-q2_0.gguf --out mtp-orca\rt
copy data\draft_vocab.bin mtp-orca\rt\
```

Put `packs\` and the GGUF on the fastest drive you have: the start is a 63 GiB read.

## Run

One-shot:

```bat
build\strata.exe --pack packs\orca-nvfp4 --native models\orca-nvfp4.gguf --native-dense-gguf models\orca-nvfp4.gguf ^
  --ple-gguf models\ple-fp8.gguf --embd-gguf models\token-embd-bf16.gguf ^
  --mtp mtp-orca\rt --spec 4 --spec-min-p 0.5 --prefill auto ^
  --expert-profile data\expert-profile.bin --expert-cache auto ^
  --max-context 262144 --kv int8 --tokens-file prompt.txt --max-new 256
```

OpenAI-compatible server: save the same arguments as a config (`{"exe": "build/strata.exe", "args": [...],
"tokenizer": "packs/orca-nvfp4/tokenizer", "host": "127.0.0.1", "port": 8097}`) and start
`python -m serve.server --engine strata --config that.json`.

`--native` and `--native-dense-gguf` both point at the GGUF: `--native` alone would take the PLE file for a second
shard of the same model.

## Claude Code

```bat
set ANTHROPIC_BASE_URL=http://127.0.0.1:8097
set ANTHROPIC_API_KEY=local
set ANTHROPIC_MODEL=strata-nvfp4
set ANTHROPIC_SMALL_FAST_MODEL=strata-nvfp4
claude
```

The model is hybrid (Gated DeltaNet layers keep a recurrent state that cannot be cut back to a position), so an
agent turn that differs from the last one a few tokens in would re-read everything without checkpoints. The server
keeps one at every turn boundary. Measured with a real `claude -p` session doing three tool turns: the first request
read its 21,964-token system prompt once (5.8 s), the next ones 209 and 106 new tokens (0.55 s and 0.4 s). An edit
in the middle of a history falls back to the checkpoint just before it. `/v1/messages/count_tokens` is served, and
a request that asks for no thinking (Claude Code's small helper calls) gets none.

## Large pages (why a reboot)

This is not more memory, it is bigger pages. Windows maps memory in 4 KB pages, so the 63 GiB expert arena is
16.5 million of them; the CPU part of every token reads experts from it at DRAM speed, and each page needs a TLB
entry. With 2 MB pages it is 32 thousand, and the CPU pool holds steady: 7.3-7.7 ms per round on this machine
against 7.4-11 ms with 4 KB pages. Decode itself is GPU-bound here, so the rate moves little; the point is the
steadiness (and a slightly faster start and exit).

Windows only gives large pages to an account that holds *Lock pages in memory* (SeLockMemoryPrivilege), and it
puts a privilege into a sign-in's token only when that sign-in starts. So:

1. `powershell -ExecutionPolicy Bypass -File tools\enable-large-pages.ps1` — asks for admin (UAC) and grants it to
   the current user through `secedit` (works on Windows Home, which has no `secpol.msc`); the policy as it was is
   saved to `%LOCALAPPDATA%\strata-large-pages\before.inf`, and `-Revoke` takes it back.
2. **Sign out and back in, or reboot.** Locking the screen is not a new sign-in.
3. Check: the same script with `-Check`, or the engine's start line `expert arena: ... large pages (2097152 B)`.

Without the privilege the engine says `large pages refused ... VirtualAlloc error 1314` and runs on 4 KB pages.
Error 1450 instead means the privilege is there but Windows found no 32 thousand free 2 MB blocks (memory
fragmented after a long uptime): it falls back the same way, and a reboot clears it. The arena is locked in RAM
either way - CUDA pins it for the GPU's copies.

## Switches

| | |
| --- | --- |
| `STRATA_PREFILL_NVFP4=w4a8\|w4a4\|fp16` | prompt path precision (default `w4a8`) |
| `--embd-gguf PATH` | the token embedding from this GGUF (BF16 from `tools/embd_bf16_pack.py`) |
| `STRATA_PREFILL_BF16X2=2\|1\|0` | exact inputs to the prompt path's BF16 projections: all but the hyper-connection (default), all (~9% slower prompt reading), off |
| `STRATA_KV_ROT=0` | int8 K/V without the Hadamard rotation (A/B) |
| `STRATA_ROPE_LEGACY=1` | the old float fast-math RoPE angles (A/B) |
| `STRATA_COMMIT_SYNC=1` | the verify commit waits for its graph again (A/B) |
| `--pcie-frac F` | share of cache misses fetched over PCIe (NVFP4 default 0.25) |
| `STRATA_NO_LARGEPAGES=1` | 4 KB pages even when large pages are allowed (A/B) |
| `STRATA_NO_NVFP4_512=1` | CPU pool on ggml-cpu's NVFP4 dot instead of the AVX-512 rows |
| `STRATA_BUFFERED_LOAD=1` | the original buffered arena loader |
| `STRATA_VERIFY_ARENA=1` | print a checksum of the loaded arena |
| `STRATA_DUMP_FIRST_LOGITS=file` | write the first generated token's logits (compare prompt paths) |
| `STRATA_DUMP_MOE_INPUT=file`, `STRATA_DUMP_MOE_LAYER=l` | dump one layer's real MoE input rows |
| `STRATA_REQUEST_LINES=1` | `serve.server` echoes one summary line per request to stdout |

## Tests

| executable | checks |
| --- | --- |
| `nvfp4_avx512_parity [experts.bin]` | AVX-512 rows vs ggml-cpu (`--expert`: one whole expert vs FP64; `--bw`: DRAM rate) |
| `nvfp4_expert_gpu_parity experts.bin` | the GPU decode path's experts vs FP64, one per sampled layer |
| `mmq_nvfp4_parity experts.bin` | MMQ products vs FP64 (`--group`, `--layers`, `--real dump layer`) |

## Limits

- Built and measured on Windows with one sm_120 GPU only; the MMQ and AVX-512 paths assume a Blackwell card and an
  AVX-512 CPU (both fall back where the code allows, but other setups are untested).
- A decode round (~21 ms) is the GPU running back to back, ~4.6 ms of it pulling the PCIe share of the experts
  and ~3.4 ms waiting for the CPU's share, which reads DRAM at ~55 of the ~65 GB/s this platform does. More VRAM
  for the expert cache or more memory bandwidth are what would move it; docs/NVFP4.md lists what was tried.
- The model is an abliterated fine-tune: it does not refuse. What it is used for is on whoever runs it.

## Releasing

```bat
python release\make_windows_bundle.py      :: clean tree only; builds build-release\ itself, zips, SHA-256
git push origin HEAD:main
python release\publish.py --title "Strata NVFP4 v... - what changed" --notes notes.md   :: --dry-run first
```

`publish.py` names this repository in every `gh` call (a clone's gh default can point at upstream) and refuses a
zip whose `engine\BUILD.json` is not the pushed HEAD, this version and a clean tree.

## License

MIT, as upstream ([LICENSE](LICENSE), copyright Niko1221 and the Strata contributors). llama.cpp / ggml code compiled
into the build is MIT as well.

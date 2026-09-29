# Strata NVFP4

**Qwen3.8-Flash-Next (125B hybrid MoE) in NVFP4 on one RTX 5090 + 128 GB of RAM.** A fork of
[Niko1221/Strata](https://github.com/Niko1221/Strata) that runs a ModelOpt **NVFP4** checkpoint — here
[OrcaRouter's abliterated Flash-Next](https://huggingface.co/jpezzulli/OrcaRouter-Qwen3.8-Flash-Next-Uncensored-ModelOpt-NVFP4) —
instead of the Q2/Q3 quants Strata ships for. NVFP4 keeps the experts at 4.5 bits with calibrated scales, which is
why this fork exists: the 2-3 bit quants were noticeably less accurate on the same model.

Strata itself keeps the 63 GiB of routed experts in RAM, caches the most-used ones in VRAM, computes the misses on
the CPU and over PCIe in parallel with the GPU, and decodes with an MTP draft head. Everything about that design is
upstream's; the original README is kept as [README.upstream.md](README.upstream.md).

## Measured

RTX 5090 (32 GB, PCIe 5 x16), Ryzen 9 9950X3D, 128 GB DDR5-5600, Samsung 9100 PRO, Windows 11, CUDA 13.3.
262,144-token context, KV cache int8, large pages on.

| | |
| --- | ---: |
| Writes answers, short chat | ~117 tokens/s |
| Writes answers, 32K context | ~124 tokens/s |
| Reads a 32K prompt | ~5,200 tokens/s (time to first token 7.5 s) |
| Start to ready | ~10 s (63 GiB of experts read at ~11 GiB/s) |

Prompt precision, first-token KL divergence against the FP16 reference over 8 prompts (1K-8K tokens):

| prompt path | KL mean | top-1 agreement | prompt speed (4-8K) |
| --- | ---: | ---: | ---: |
| `w4a8` (default): int8 tensor cores, 8-bit activations | 0.0018 | 8/8 | 2,503 tok/s |
| `w4a4`: Blackwell FP4 x FP4 | 0.0080 | 7/8 | 2,904 tok/s |
| `fp16`: dequantize + FP16 GEMM (reference) | — | — | 1,905 tok/s |
| noise floor (same FP16 path, other chunking) | 0.00023 | 8/8 | |

Method, per-product error measurements and everything that was tried and dropped: [docs/NVFP4.md](docs/NVFP4.md).

## What this fork adds

- **NVFP4 routed experts end to end.** `tools/nvfp4_convert.py` repacks the ModelOpt checkpoint into a GGUF with
  llama.cpp's own converter (lossless; each expert's `weight_scale_2` kept), and `tools/iq_pack.py` writes each
  expert blob with a 16-byte tail `{s_gate, s_up, s_down, 0}`. CPU pool, GPU decode, FP16 and MMQ prompt paths all
  read it; each global scale multiplies its own projection's FP32 output, as llama.cpp applies `.scale`.
- **W4A8 prompt path for Blackwell.** llama.cpp's MMQ builds NVFP4 only as FP4 x FP4 on sm_120, rounding the
  activations to 4 bits too. `src/prefill/mmq_nvfp4_w4a8.cu` compiles its int8 path for this GPU instead: 16x
  smaller error per product (0.46% vs 7.2% on real activations) at 86% of the FP4 speed. `STRATA_PREFILL_NVFP4`
  switches between `w4a8`, `w4a4` and `fp16`.
- **The n-gram (PLE) table as shipped.** Qwen stores the 51.2e9-value table in FP8 E4M3; Strata read it only as
  IQ4_NL, 8.1% off per row. `tools/ple_fp8_pack.py` keeps the FP8 bytes unchanged and the engine reads either format.
  The IQ4_NL table moved the first token's distribution by KL 0.0026 on average - more than the whole W4A8 prompt
  path - at the same speed (the table is read from the SSD, 16 rows a token).
- **AVX-512 NVFP4 rows for the CPU pool** (`src/kernels/cpu/nvfp4_avx512.cpp`): each 64-value block decoded once
  per verify window, same arithmetic as ggml-cpu (1.8x at one token, 3.7x at seven).
- **Loading.** `experts.bin` is read unbuffered straight into the pinned arena by 16 readers while another thread
  registers it with CUDA one layer ahead: 19 s to 5.6 s from a PCIe 5 drive.
- **Tuned for NVFP4's larger experts:** 25% of the cache misses go over PCIe (was 55%; +13% decode short, +17% at
  32K), and prompt chunks go up to 32,768 tokens (+47% prompt speed at 32K).
- **The abliterated MTP head.** `tools/mtp_extract.py` takes the draft head from the fine-tune itself, not from the
  original model, so the drafts match the model that verifies them.
- Parity tests for every new kernel and diagnostics for comparing prompt paths (below).

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

:: 3. GGUF (NVFP4 experts, Q8_0/BF16 dense) and the pack (experts.bin 63 GiB, tokenizer)
.venv\Scripts\python tools\nvfp4_convert.py --model models\orca-nvfp4 --outfile models\orca-nvfp4.gguf
.venv\Scripts\python tools\iq_pack.py --gguf models\orca-nvfp4.gguf --out packs\orca-nvfp4

:: 4. the fine-tune's own MTP draft head
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
  --ple-gguf models\ple-fp8.gguf ^
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
- Decode is GPU-bound on this box; the attention, hyper-connection and dense Q8_0 kernels are close to the card's
  bandwidth, so there is little left there without lower-precision weights.
- The model is an abliterated fine-tune: it does not refuse. What it is used for is on whoever runs it.

## License

MIT, as upstream ([LICENSE](LICENSE), copyright Niko1221 and the Strata contributors). llama.cpp / ggml code compiled
into the build is MIT as well.

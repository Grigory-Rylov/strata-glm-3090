# NVFP4 routed experts

Measured on Windows 11 with an RTX 5090 (32 GB, sm_120), Ryzen 9 9950X3D and 128 GB DDR5-5600, 2026-09-29.

The target is a ModelOpt NVFP4 checkpoint of Qwen3.8-Flash-Next (OrcaRouter's abliteration,
`jpezzulli`'s quantization). `tools/nvfp4_convert.py` repacks it without loss into a GGUF whose routed
experts are NVFP4 (64-value blocks: four UE4M3 sub-block scales and 32 bytes of E2M1 codes) and whose
per-expert `weight_scale_2` lands in `blk.N.ffn_{gate,up,down}_exps.scale`. `tools/iq_pack.py` writes
each expert blob as `[gate | up | down]` plus a 16-byte tail `{s_gate, s_up, s_down, 0}`.

## Where the global scales go

Each scale multiplies its own projection's FP32 output, as llama.cpp applies `.scale`:
`h = silu(s_gate * (G x)) * (s_up * (U x))`, `y = s_down * (D h)`.

The scales are ~1e-4 each. An earlier revision folded `s_down` into up, which is algebraically the same
but leaves the hidden near 1e-5. The hidden is then rounded to Q8_0 (CPU) or q8_1 (GPU) blocks whose
scale is FP16, and `amax / 127` lands deep in FP16's subnormals: one expert's output was off by 2-12%
instead of 1.1% (`nvfp4_avx512_parity --expert`). The FP16 prompt path had the same problem in its
dequantized up weights (~1e-8). Any future fold must be checked against FP16's range.

## Prompt path precision

`STRATA_PREFILL_NVFP4` picks how the batched prompt path multiplies the experts:

| mode | kernel | activations | one product vs FP64 (`mmq_nvfp4_parity`) |
|---|---|---|---|
| `w4a8` (default) | llama.cpp MMQ, int8 tensor cores | q8_1, float block scales | 0.53% |
| `w4a4` | llama.cpp MMQ, Blackwell FP4 MMA | NVFP4 | 8.6% |
| `fp16` | dequantize + cuBLAS FP16 GEMM | FP16 | reference |

On sm_120 mmq.cuh compiles NVFP4 only as FP4 x FP4. `src/prefill/mmq_nvfp4_w4a8.cu` compiles its int8
path for this GPU by hiding Blackwell's FP4 MMA from both the device code and the host's tile config;
decode already multiplies at this precision.

First-token KL divergence against `fp16`, 8 prompts (code and prose, 1K-8K tokens); the noise floor is the
same `fp16` path cut into 4096-token chunks instead of 8192 (another summation order, same arithmetic):

| mode | KL mean | KL median | KL max | top-1 agreement | prefill, 4-8K prompts |
|---|---|---|---|---|---|
| noise floor | 0.00023 | 0.00003 | 0.0016 | 8/8 | - |
| `w4a8` | 0.0018 | 0.0010 | 0.0096 | 8/8 | 2,503 tok/s |
| `w4a4` | 0.0080 | 0.0027 | 0.040 | 7/8 | 2,904 tok/s |
| `fp16` | reference | | | | 1,905 tok/s |

`w4a8` keeps the default: 4.5x closer to the reference than `w4a4` at 86% of its speed, and the precision
every generated token already runs at. `fp16` is the choice when the prompt must be read exactly.

On a real prompt's rows (`STRATA_DUMP_MOE_INPUT` + `mmq_nvfp4_parity --real`, layer 20, its 12 busiest
experts, max|x|/rms 4 median and 9 at most) one product is off by: `w4a8` 0.46% (gate/up) and 0.90% (down, whose
SwiGLU input has the heavier tails), `w4a4` 7.2% and 8.4%, `fp16` 0.017% and 0.021%.

## The n-gram (PLE) table

Layer 1 adds 16 rows of a 320,001,536 x 160 n-gram table per token. Qwen ships it in FP8 E4M3 (128 shards of
[2500012, 160] and one scale, 51.2 GB); no source holds more precision (OrcaRouter's BF16 copy is this FP8
widened). Strata read only IQ4_NL (ISTA-DASLab's shard 2, 28.8 GB), which is 8.1% off the FP8 values per row -
correlation 0.996-0.997 on rows from every shard, so the same table in the same order, and the abliteration left it
alone. `tools/ple_fp8_pack.py` copies the FP8 bytes into a GGUF (I8, `strata.ple.format` = f8_e4m3,
`strata.ple.scale`); `ple_fp8_parity` checks the engine's rows against torch's decode of the checkpoint, bit for bit.

| first-token KL, 8 prompts (1K-8K) | mean | median | max | top-1 |
| --- | ---: | ---: | ---: | ---: |
| noise floor (summation order) | 0.00023 | 0.00003 | 0.0016 | 8/8 |
| w4a8 prompt path vs fp16 | 0.0018 | 0.0010 | 0.0096 | 8/8 |
| **IQ4_NL PLE vs FP8 PLE** | **0.0026** | **0.0012** | **0.013** | 8/8 |

It costs 22 GB more disk and nothing else: the table stays on the SSD (16 page reads a token either way; a row is
160 B instead of 90) and the row cache grows from ~95 to ~160 MB. Prompt reading speed was unchanged.

## CPU experts

`src/kernels/cpu/nvfp4_avx512.cpp` computes the CPU pool's NVFP4 rows in 512-bit lanes: each 64-value
block is decoded once for every token of a verify window (vpdpbusd on |E2M1| codes with the sign moved
onto the activation). Same arithmetic as ggml-cpu's `ggml_vec_dot_nvfp4_q8_0`, only the float additions
are ordered differently (worst 4.5e-7 relative, `nvfp4_avx512_parity`). `STRATA_NO_NVFP4_512=1` falls
back to ggml-cpu.

| tokens | ggml-cpu | AVX-512 | speedup |
|---|---|---|---|
| 1 | 0.102 ms | 0.058 ms | 1.76x |
| 4 | 0.410 ms | 0.125 ms | 3.29x |
| 7 | 0.719 ms | 0.196 ms | 3.66x |

(one expert's gate+up, 1280 rows of 2560, cache-resident; the pool itself is bound by DRAM)

## Loading

`experts.bin` is read unbuffered (`FILE_FLAG_NO_BUFFERING`) straight into the arena by 16 readers, while a
second thread registers the arena with CUDA one layer ahead of them (`PinnedArena::Deferred` +
`register_slices`): registering 63 GiB of 4 KiB pages alone takes 6.6 s, and the buffered reader it replaces
(`STRATA_BUFFERED_LOAD=1`, the A/B arm) copied through the file cache at ~3.3 GiB/s wall. From a PCIe 5 drive
(13.4 GiB/s unbuffered) the arena now loads in 5.6-6.0 s instead of 19 s, and the session is up in ~10 s
instead of 25. `STRATA_VERIFY_ARENA=1` prints a checksum of the loaded arena; both loaders give the same one.

Decode A/B, 300 tokens, 262K context (median of 2): pipelined per-layer registration 103.5 tok/s, whole-arena
registration 103.2; ggml-cpu's NVFP4 rows 105.3 (the pool is DRAM-bound, so the kernel above moves its time by
2-3% and the decode rate not at all).

## Tuning measured on this machine (after the fixes above)

Kept:
- **PCIe share 0.25** (was 0.55): the copy kernel fetching the PCIe share was 40% of GPU kernel time and sits on
  the GPU's critical path. Decode 103 -> 117 tok/s short, 105 -> 124 at 32K.
- **Prefill auto chunks up to 32768**: a 32K prompt reads at 5201 tok/s instead of 3535 (TTFT 10.1 -> 7.5 s).
- **Large pages** once the account holds SeLockMemoryPrivilege (needs a fresh logon): the CPU pool holds
  ~7.5 ms/round where 4 KB pages wandered 7.4-11; decode itself is GPU-bound, so the rate barely moves.

Tried and dropped (no gain, or worse):
- DMA copies for the PCIe share (`--pcie-mode dma`) at 0.25 and 0.40: ~32 rounds/s either way at 32K.
- `--spec 5/6`, `--spec-min-p 0.3/0.7`: longer windows accept more but cost more; 0.3 is 15% slower.
- More pool workers (23, 31) or other prefetch distances: the pool sits at ~55 GB/s from DDR5-5600.
- An expert profile ranked by this model's own routing (6 prompts x 1000 tokens): it covered 69% of the
  traced routing against 40% for the shipped profile, but missed MORE on held-out prompts (CPU experts per layer
  5.3 vs 4.0 short, 10.8 vs 6.6 at 32K) - it fits the traces; the shipped profile generalises.
- A q4_0 / q8_0 MTP head: tools/mtp_pack.py writes them, but the drafter only runs Q2_0 experts.

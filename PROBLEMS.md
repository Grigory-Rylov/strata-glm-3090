# PROBLEMS — strata-glm на Linux / RTX 3090 (ветка `3090`)

Порт и проверка движка GLM-5.3-Flash на железе, отличном от целевого (цель: Windows, RTX 5090 32 GB, sm_120).
Все цифры — литеральный вывод команд 2026-10-02/03 на хосте `epyc-ai`; ничего не оценено, если не помечено PROVISIONAL.

## Стенд (замер)

| | |
|---|---|
| GPU | 3× RTX 3090 24 GiB, **sm_86**; прогоны — GPU2 (свободная); GPU0 держит `ninfer-serve`, GPU1 — движок Strata/Qwen3.8, GPU2 держал ComfyUI (остановлен 03.10 для гейта) |
| RAM | 251 GiB |
| сборка | CMake Release, `CMAKE_CUDA_ARCHITECTURES=86`, nvcc 13.2 (`/usr/local/cuda-13.2`), gcc 13.3, ядро 6.8 (Ubuntu 24.04) |
| база | `sergqwer/strata-glm @ed37419` + четыре патча (`../408032gb/patches/strata-glm/000{1,2,3,4}`), применены как коммиты 4b92e97, 62d2222, 8aac492, 9d1755a |

## Что работает (замер)

1. **Порт на Linux (0001+0003)** собирается без новых предупреждений; `DiskReader` через `O_DIRECT pread`
   прошёл функциональный тест на реальных файлах
   (`../408032gb/tests/strata_glm/run_diskreader_test.sh`): **ALL OK (0 failures)** — зеркала, не выровненные
   смещения и назначения, хвост файла, приоритеты, отсутствующий файл.
2. **Сквозной инференс на GPU2**: 40 prompt-токенов + 8 decode, `exit=0`, сгенерированы токены
   (`output : 11221 24780 23545 9028 1136 11930 1146 21`).
3. **G4, бит-экзакт сплит на одной карте** (п. 2 README патчей): `--layer-split 14,25,35 --gpus 0,0,0,0` —
   `ref.f32` и `split.f32` (619 520 B логов) **совпали побайтно**, токены идентичны. Логиты совпали — вопрос
   «не зависит ли результат ядра от того, откуда прочитан блоб» закрыт положительно.
4. **Валидация аргументов G4**: `--gpus` не по числу стадий, невозрастающий `--layer-split`, разрез вне
   1..44 — каждый случай `exit=2` с внятным сообщением.
5. **Быстрый путь (ярусы VRAM+RAM) на Linux — портирован 03.10 (патч 0004)**: `TieredExpertSource` переведён
   на posix (`O_DIRECT pread` окна, буферные `pread`-ридеры, общее тело `load/stream`); ярус 10 755 экспертов
   (141.91 GiB, 43/43 slice pinned) загружается за 172.5 s; **бит-экзакт с disk-only прогоном**
   (`tiered.f32` = `ref.f32`, cmp чистый) — порт численно без потерь.

## Чем проверялось: тестовый пак

Движок не читает канонические веса (GGUF, см. проблему 1), поэтому прогоны сделаны на тестовом паке `../glm-synthetic/`
(`gen_synth.py`):

* `ckpt/weights.bin` 17.2 GB — все dense-тензоры, которые читает движок; имена и формы вынесены из
  констант и вызовов ядер `glm_main.cpp`, значения — малые случайные (bf16 ~N(0, 0.02));
* `pack/dense.txt` — таблица тензоров формата движка;
* `pack/experts.bin` 159.5 GiB sparse; MoE-слои 0 и 11 заполнены случайными блобами в нативном для
  движка формате, остальные читаются нулями;
* `profile.bin` — тестовый профиль маршрутизации для ярусного пути.

Это проверяет **все кодовые пути** (загрузка, квантизация dense, ядра KDA/DSA/MoE, стадии, чтение
experts.bin, ярус RAM) и **механику скорости** (байты, копии и ядра те же, что на любых весах), но
**не качество модели**: «качество» токенов на тестовом паке неинформативно.

## Проблемы

1. **Канонические веса — в GGUF, а движок умеет только свой pack.** Канонические веса:
   `/mnt/data/home/grishberg/models/GLM-5.3-Flash-GGUF/UD-Q4_K_XL/GLM-5.3-Flash-UD-Q4_K_XL-*-of-00006.gguf`
   (186 GiB; зафиксировано в `../408032gb/AGENTS.md`). Движок читает только pack формата движка
   (`dense.txt` + `experts.bin`) — нужен конвертер GGUF→pack (деквант Q4_K-блоков → квант в формат
   движка; формат экспертов движка = ggml block_nvfp4, тот же, что умеет llama.cpp).
   `tools/glm_pack.py` — Windows-only и ждёт другой формат входа, на этой машине не используется.
2. **24 GiB VRAM не хватает при квантизации dense по умолчанию**: OOM на `cudaMalloc` при загрузке.
   С `--dense-fp4 all` эталонный прогон оставляет 1.5 GiB свободными; 4 стадии на одной карте — 17.9 GiB
   после загрузки всех стадий (суммарно ~6 GiB). На 5090/32 GB умещается дефолт.
3. **Конкуренция за GPU**: все три карты заняты сервисами; гейт ждал освобождения GPU2 ~20 минут, пока не
   остановили ComfyUI. Без выделенной карты прогоны нестабильны по памяти.
4. **G4 на одной карте — стадии последовательны** (известное ограничение этапа 1): выигрыша по tok/s на одной
   карте нет и быть не может; проверка бит-экзакта — единственное, что она доказывает. Перекрытие стадий и
   предсказание через границу стадии — будущие патчи.
5. **sm_86 в CI нет**: `ci.yml` собирает sm_89; локальная сборка sm_86 проходит, но регулярной проверки нет.
6. ~~Быстрый путь (`--profile`, ярусы VRAM+RAM) на Linux не портирован~~ — **портирован 03.10, см. «Что
   работает» п. 5 и патч 0004**. Остаток риска: подтверждение на реальных весах (hit-rate, качество) упирается
   в вопрос 1.

## Скорости (замер, синтетика, disk-only путь, GPU2)

| прогон | prompt | decode |
|---|---|---|
| единый движок, `--chunk 0` | 40 tok / 34.4 s = 1.16 tok/s | 16 tok / 14.0 s = **1.15 tok/s** |
| 4 стадии на одной карте | 40 tok / 37.4 s = 1.07 tok/s | 8 tok / 6.9 s = **1.17 tok/s** |
| **ярус RAM (патч 0004), `--profile`, `--chunk 0`** | 1024 tok / 52.3 s = 19.58 tok/s | 32 tok / 1.6 s = **19.69 tok/s** |
| ярус RAM, `--profile`, `--chunk 512` | 1024 tok / 15.5 s = 66.18 tok/s | 32 tok / 1.6 s = 19.85 tok/s |
| ярус RAM, `--profile`, `--chunk 1024` | 1024 tok / 8.2 s = 124.18 tok/s | 32 tok / 1.6 s = 19.81 tok/s |
| ярус RAM, `--profile`, `--chunk 2048` | 1024 tok / 7.8 s = 132.02 tok/s (плато короткого промпта: упор в PCIe-стрим 171 GB на 1024 токена) | 32 tok / 1.6 s = 19.83 tok/s |
| **ярус RAM, промпт 16K, `--chunk 4096`** | 16 384 tok / 37.4 s = **437.98 tok/s** | — |
| **ярус RAM, промпт 64K, `--chunk 4096 --max-context 65536`** | 65 536 tok / 155.8 s = **420.74 tok/s** | — |

Механика disk-only: на токен decode читается 42×8×14 155 792 B = **4.76 GB** (fread из page cache + H2D,
сериализовано sync-ом на слой) → эффективные ~5.5 GB/s — потолок disk-only конвейера, не модель.
Ярусный путь (замер GLM_TIMING, `glm-synthetic/tiered.log`): токен 49.4 мс = expert copies 13.0 + expert
kernels 0.4 + остальное (dense, роутеры, sync) 36.1 мс; prefetch 100% попаданий, disk waits 0.0 мс,
экспертов с карты 99.2% (RAM->GPU 0.8%, disk 0.0%). MAP_HUGETLB на машине не настроен — ярус на 4 KB
страницах, на регистрацию не повлияло (43/43 slice pinned).
Потолок сместился с дисковой полосы на dense-вычисления — на реальных весах dense те же, оценка скорости
переносима. Ориентиры: upstream на 5090 с ярусами — 18 tok/s decode (README); ik_llama.cpp (GGUF Q4_K_XL)
на этой машине — tg 7–15 tok/s (`../408032gb/PROBLEMS.md`).

## Вопросы

1. **Pack из канонических весов.** Для G5/G6 нужен pack, сконвертированный из канонического GGUF
   (проблема 1); до этого точность и tok/s на реальных весах непроверяемы, тестовый пак закрывает
   только механику.
2. **Достаточен ли гейт G4 на тестовом паке**, или приёмка требует реальных весов (тогда вопрос 1)?
3. **Где проверять G4 на 4 картах** (`--gpus 0,1,2,3`)? На машине 3×3090, две заняты сервисами; вариант
   «3 стадии на 3 картах» тоже не соответствует плану (4×4080).
4. **Нужен ли путь GGUF → pack?** Переквант GGUF в формат движка потеряет точность; альтернатива — учить
   движок читать нативный GGUF-формат экспертов, это отдельный большой этап.
5. **Добавить ли job sm_86 в CI** (сборка + `run_diskreader_test.sh`)?
6. ~~Портировать ли `TieredExpertSource` (ярус RAM) на Linux~~ — **сделано 03.10 (патч 0004, 19.65 tok/s
   decode на тестовом паке)**. Осталось: подтвердить на реальных весах (вопрос 1).
6. **Портировать ли `glm_pack.py` на Linux** заранее, до решения по вопросу 1?

## 2026-10-03: real weights - the routing spread is the decode ceiling

The NVFP4 checkpoint packed (glm_pack.py, 42 layers, 171.2 GB, 24 experts verified value by
value, 34 min). Routing profile from 16 real segments (code/docs/prose, glm_profile.py):
expert reuse from the previous token 24.6% (the synthetic pack: ~60%), per-layer top-1310
covers 42-64% of the routed mass. First run on the real pack (GPU2, chunk 4096, 16K prompt,
256 new tokens, tier 562 VRAM + 11036 RAM + 498 disk):

  prefill 381.59 tok/s (synthetic 437.8), decode 9.25 tok/s (synthetic 19.80)
  decode token 106.5 ms: copies (with disk waits) 64.1, kernels 4.8, the rest 37.6

The decode gap is the 498 disk-side experts the spread routing keeps hitting (~4.5 disk
experts/token at 14.16 MB each). The RAM tier already holds 91% of all experts - the fix is
not a bigger tier, it is either (a) more VRAM (layer-split: 2 cards = 2x the hot set),
(b) speculative prefetch of the router's next-token candidates, or (c) the ik_llama-style
GGUF path where all experts live in RAM (no disk tier at all). MAP_HUGETLB unavailable on
this host (no hugetlb pool) - 4 KB pages, minor.

## 2026-10-03: real weights answer - quality confirmed, decode is PCIe-bound

Quality: "What is the capital of France? Answer in one short sentence." ->
"<|im_start|>The capital of France is Paris." (stops at EOS). A 256-token continuation of a real
16K-token docs+code prompt produces fluent on-style API documentation. The pack is value-true.

Speeds on the real pack (GPU2, chunk 4096): prefill 16K real text 401.49 tok/s (random ids:
381.59); decode 6.85 tok/s on real text, 9.25 on random ids, 8.03 on the 13-token prompt.
The decode token at 144.1 ms: copies 99.5 (48.6% of expert loads come from RAM = ~2.3 GB/token
over the ~25 GB/s PCIe), kernels 7.0, the rest 37.6. Disk is NOT the ceiling on real text
(0.1%, 1.4 ms) - the ceiling is RAM->GPU expert copies. Layer-split doubles the VRAM-resident
hot set (51.3% VRAM now) and adds a second PCIe line: the measured case for --gpus 2,X.

## 2026-10-03: decode 6.85 -> 13.89 - the CPU pool is the lever, not PCIe tuning

The decode ceiling was RAM->GPU copies (2.3 GB/token at 23 GB/s, ~90% of PCIe). The fix is not
to move those bytes faster but to not move them: `--cpu-share 1.0 --cpu-threads 90` computes the
RAM-resident experts on the 96-core EPYC (the pool already existed for cache misses; the share
knob routes RAM hits to it). Copies 99.5 -> 8.5 ms; decode 6.85 -> 13.30 tok/s (256 steps, real
16K prompt). `--dense-i8` (int8 dense GEMV) cuts the dense part 37.6 -> 27.3 ms: 13.89 tok/s
(64 steps). The breakdown stays ADDITIVE (copies + kernels + dense): the per-token chain
dense -> x to host -> CPU experts -> back is a dependency, nothing overlaps - speculative
verification amortizes the fixed 27-37 ms dense part over k drafted tokens, the next lever.

Tuning table (real pack, 16K prompt, GPU2): baseline 6.85 | share 1.0/90 13.30 | share 0.7/90
10.30 (copies return, CPU idle - worse) | share 1.0/90 + latent-i8 12.82 (no gain) | share
1.0/90 + dense-i8 13.89 BEST. Bugs found: --cpu-threads >= nproc (96, 120) HANGS decode after
prefill; --latent-i8 + cpu-share hangs at 96 threads, fine at 90. VRAM-set A/B exhausted:
chunk 2048 grows the VRAM set 562 -> 690 but prefill drops 401 -> 241 and decode does not move
(the hit rate is set by routing concentration, not the tail); --vram-experts 650/700 OOM the
chunk-4096 prompt buffers (the auto-size is correct).

## 2026-10-03: --spec K - speculative decoding, and the MoE ceiling on it

The lever named above, built. `--spec K` drafts up to K tokens by prompt lookup (the longest suffix of
the sequence, 2..8 tokens, that occurs earlier in it, plus what followed that occurrence), runs [the
token just taken, the drafts] through the prompt path at the current position and reads the head at
every row. Each position's token is the model's own argmax at that row, so the drafts choose only
which positions are computed together, never what comes out - greedy by construction. `--spec 0` is
the old loop, unchanged (verified: same 64 ids, same speed). `--spec` asks for `--profile`: without the
tiers a chunk re-reads all 288 experts of a layer, so a pass over T of them would save nothing.

What a pass leaves behind is the KDA layers' recurrence, advanced by K+1 tokens when only a prefix was
accepted. Each pass snapshots S, the conv state, the raw q/k/v and the post-prep scan inputs of all 34
KDA layers; spec_fix(keep) restores S and the conv, re-runs the conv and the scan over the kept rows.
The MLA, indexer and pool caches need no repair: their garbage rows sit at or beyond the next pass's
first position, and a block is first read at query >= 4j+4, always after the pass that writes row 4j+3
has re-pooled it.

Three pieces the pass needed that the engine did not have. CpuExperts::start_multi computes the RAM
experts on the CPU pool over all their rows at once - an expert's blob is read once for its rows; the
alternative, a PCIe copy of ~20 blobs a layer, is ~500 ms a pass. dmat runs the dense projections of a
pass through gemv: cuBLASLt dequantizes each quantized matrix to BF16 on every call, bytes with
nothing to do with T and a rounding the decode token never sees, while gemv reads the quantized weights
once for all its rows as the decode path does. That was 370 -> 205 ms a pass, and acceptance 64 -> 83%
in the short high-copy stretch, because verifier and decoder now round alike. And the pool's job boundary
had to be closed: a worker leaves its gate the moment the generation changes but is still inside its
claim loops when the last unit lands, so the producer could reset the counters and the shape under a
straggler - it then takes a unit of the next job with the previous job's n_ and pointers, which either
sets finished_ early (half-computed rows copied to the GPU) or never sets it (a permanent spin in
cpux.wait()). Harmless while every job had one row; --spec alternates single-row and multi-row jobs
every iteration. start()/start_multi() now wait for every worker back at the gate before publishing.

Measured (real pack, GPU2, 16K prompt, 256 tokens, share 1.0/90, dense-i8, one batch, same build):
spec 0 13.60 tok/s (a step 71 ms, 50.8% of the experts in VRAM) | spec 3 14.12 (55.6% of the drafts
accepted, 2.67 tokens a pass, a pass 206 ms, break-even 2.76, 44.5% VRAM) | spec 5 10.47 (46.7%, 3.33
tokens, a pass 328 ms, break-even 4.0, 44.4% VRAM). The same build measured spec 0 at 14.61 in another
run, so the spread between runs is as wide as the difference between the modes - and the pass machinery
is not free: its staging and snapshot buffers cost ~650 MB of VRAM the expert tier would otherwise
hold, six points of the hot set. The cost model is a pass = F + V*T with F ~ 100 ms (dense, attention,
the GPU's experts - read once for all T rows) and V ~ 25 ms a token (the CPU pool's MACs and the extra
expert blobs, both linear in tokens) against a 66-82 ms step. The first 16 tokens, deep inside a block
the model is copying, ran 15.41 tok/s at 83% acceptance; the rest of the run copies less.

The MoE is the ceiling, not the plumbing. 8 of 288 experts a token, so a pass over T tokens touches
~6.5T distinct experts: the CPU pool reads ~311 MB a layer where a decode step reads ~85 MB. Expert
bytes are ~70% of a token's cost and barely amortize; only the dense part and the per-step chain do,
which bounds the gain near 1.2x even at perfect acceptance. Speculative decoding on this engine is
marginal by construction, not by tuning.

So the flag measures itself rather than trusting a constant: every 8th position (until six plain
samples) goes through a plain step, and once both sides have samples, tokens a pass is compared with
the measured pass/step ratio and speculation stops when it is below it - `spec 3 off after 9 passes:
2.67 tokens a pass for 206.2 ms, a plain step is 74.8 ms`. Being wrong about a corpus then costs the
passes before the guard fires, not the run. Once off it stays off: acceptance decays as the model
stops copying, and re-probing costs a pass each time.

Numerics: a pass is the prompt path (FP32 experts on the CPU pool, gemv dense), a step is the decode
path (quantized experts on the GPU). The two diverge after ~15 ids here. That is legal for greedy
speculative decoding - every token of a --spec run is the argmax of a pass, and the passes are
self-consistent - but --spec K does not reproduce --spec 0's text. The straggler race above is the
likeliest explanation of the one hang seen while building this: a run stopped after the prefill with the
GPU at 0% and no reads from the process, and did not reproduce with identical flags afterwards.

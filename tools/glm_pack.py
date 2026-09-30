"""tools/glm_pack.py - a Strata pack for GLM-5.3-Flash NVFP4 (ModelOpt), straight from the checkpoint's safetensors.

    python tools/glm_pack.py --model <GLM-5.3-Flash-NVFP4 dir> --out <pack dir> [--verify 24]

Writes, one expert at a time (flat memory):

  experts.bin          the 42 MoE layers' (3..44, numbered 0..41 here) 288 experts, each a blob
                       [gate rows | up rows | down rows | {s_gate, s_up, s_down, 0}] in ggml's block_nvfp4 row format
                       (64 values: four UE4M3 scales, then 32 bytes of E2M1 codes, byte j = value j | value j+8 << 4
                       per 16-value sub-block) - the layout native_fmt(40, 40, 4096, 2048) expects, 14,155,792 B.
                       The repack is llama.cpp's _nvfp4_pack; the codes and scales are the checkpoint's bits.
  native_experts.txt   the layout, with the widths in its header (n_expert 288, n_embd 4096, n_ff 2048).
  dense.json           every other tensor of the language model (the trunk and the MTP layer 45) with its file,
                       byte offset, dtype and shape: the engine reads them from the checkpoint itself.

Resumable: experts.bin grows a whole layer at a time; a run finds the layers already written and continues.
The process stops itself above --max-private-gb of committed memory (default 8).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import pathlib
import random
import struct
import sys
import time

import numpy as np

N_LAYERS, DENSE_LEAD, N_EXPERT, N_EMBD, N_FF = 45, 3, 288, 4096, 2048
NVFP4 = 40
GU_ROW, D_ROW = (N_EMBD // 64) * 36, (N_FF // 64) * 36
BLOB = 2 * N_FF * GU_ROW + N_EMBD * D_ROW + 16
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
KV_FP4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)   # ggml: doubled


def private_gb() -> float:
    """This process's committed memory (Windows PrivateUsage), GB."""
    class PMC(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + \
                   [(n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                  "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                  "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
                                                  "PrivateUsage")]
    pmc = PMC()
    pmc.cb = ctypes.sizeof(PMC)
    k32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
    k32.GetCurrentProcess.restype = ctypes.c_void_p          # a 64-bit pseudo-handle: not a C int
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    if not psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
        raise OSError("GetProcessMemoryInfo failed")
    return pmc.PrivateUsage / 1e9


class Checkpoint:
    """The safetensors files: headers parsed once, tensors read as zero-copy views of a memmap."""

    def __init__(self, d: pathlib.Path):
        self.d = d
        self.where = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
        self.headers: dict[str, tuple[dict, int]] = {}
        self.maps: dict[str, np.memmap] = {}

    def header(self, f: str):
        if f not in self.headers:
            with open(self.d / f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.headers[f] = (json.loads(fh.read(n)), 8 + n)
        return self.headers[f]

    def info(self, name: str):
        f = self.where[name]
        h, base = self.header(f)
        t = h[name]
        return f, base + t["data_offsets"][0], t["data_offsets"][1] - t["data_offsets"][0], t["dtype"], t["shape"]

    def raw(self, name: str) -> np.ndarray:
        f, off, n, _, _ = self.info(name)
        if f not in self.maps:
            self.maps[f] = np.memmap(self.d / f, dtype=np.uint8, mode="r")
        return self.maps[f][off:off + n]


def pack_rows(w: np.ndarray, s: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """ModelOpt NVFP4 (u8 [rows, cols/2] nibbles, e4m3 [rows, cols/16]) -> block_nvfp4 rows (llama.cpp _nvfp4_pack)."""
    nb = cols // 16
    w = w.reshape(rows, nb, 8)
    vals = np.stack([w & 0x0F, w >> 4], axis=-1).reshape(rows, nb, 16)
    d = (s.reshape(rows, nb) & 0x7F).astype(np.uint8)
    qs = (vals[:, :, :8] | (vals[:, :, 8:] << 4)).astype(np.uint8)
    ns = nb // 4
    return np.concatenate([d.reshape(rows, ns, 4), qs.reshape(rows, ns, 32)], axis=-1).reshape(rows, ns * 36)


def e4m3(b: np.ndarray) -> np.ndarray:
    """E4M3FN bits -> float32 (the checkpoint's own decode, independent of the repack)."""
    b = b.astype(np.int32)
    sign = np.where(b & 0x80, -1.0, 1.0)
    e, m = (b >> 3) & 0xF, b & 0x7
    v = np.where(e == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (e - 7))
    return (sign * v).astype(np.float32)


def decode_ckpt(w: np.ndarray, s: np.ndarray, rows: int, cols: int) -> np.ndarray:
    codes = np.stack([w & 0x0F, w >> 4], axis=-1).reshape(rows, cols)
    return E2M1[codes] * np.repeat(e4m3(s.reshape(rows, cols // 16)), 16, axis=1)


def decode_blocks(raw: np.ndarray, rows: int, cols: int) -> np.ndarray:
    """block_nvfp4 rows -> float32, as ggml decodes them (doubled E2M1 values, the UE4M3 scale halved)."""
    b = raw.reshape(rows, cols // 64, 36)
    d = e4m3(b[:, :, :4]) * 0.5                                          # [rows, ns, 4]
    qs = b[:, :, 4:].reshape(rows, cols // 64, 4, 8)
    codes = np.concatenate([qs & 0x0F, qs >> 4], axis=-1)                 # [rows, ns, 4, 16]
    return (KV_FP4[codes] * d[..., None]).reshape(rows, cols)


def expert_blob(ck: Checkpoint, layer: int, e: int, verify: bool = False) -> bytes:
    pre = "model.language_model.layers.%d.mlp.experts.%d." % (layer, e)
    parts, tail = [], []
    for proj, rows, cols in (("gate_proj", N_FF, N_EMBD), ("up_proj", N_FF, N_EMBD), ("down_proj", N_EMBD, N_FF)):
        w, s = ck.raw(pre + proj + ".weight"), ck.raw(pre + proj + ".weight_scale")
        s2 = float(np.frombuffer(ck.raw(pre + proj + ".weight_scale_2").tobytes(), dtype=np.float32)[0])
        if not np.isfinite(s2) or s2 <= 0:
            raise ValueError("%s%s: weight_scale_2 %r" % (pre, proj, s2))
        packed = pack_rows(np.asarray(w), np.asarray(s), rows, cols)
        if verify:   # every value of this projection, through both decodes
            a, b = decode_ckpt(np.asarray(w), np.asarray(s), rows, cols), decode_blocks(packed, rows, cols)
            if not np.array_equal(a, b):
                raise ValueError("%s%s: the repacked blocks decode differently (max |d| %g)"
                                 % (pre, proj, float(np.abs(a - b).max())))
        parts.append(packed.tobytes())
        tail.append(s2)
    blob = b"".join(parts) + np.array(tail + [0.0], dtype=np.float32).tobytes()
    assert len(blob) == BLOB, len(blob)
    return blob


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", type=int, default=24, help="experts whose every value is checked through both decodes")
    ap.add_argument("--max-private-gb", type=float, default=8.0)
    ap.add_argument("--layers", type=int, default=N_LAYERS - DENSE_LEAD, help="MoE layers to write (tests)")
    a = ap.parse_args()
    ck = Checkpoint(pathlib.Path(a.model))
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    layer_bytes = BLOB * N_EXPERT
    n_moe = min(a.layers, N_LAYERS - DENSE_LEAD)

    # dense.json: every non-expert tensor of the language model (trunk and MTP layer), read in place
    dense = {}
    for name in ck.where:
        if ".mlp.experts." in name and ".layers.%d." % N_LAYERS not in name:
            continue   # the trunk's routed experts go to experts.bin; the MTP layer's (BF16) stay in place
        if "visual" in name:
            continue   # the vision tower is not part of the language model
        f, off, n, dt, shape = ck.info(name)
        dense[name] = {"file": f, "offset": off, "bytes": n, "dtype": dt, "shape": shape}
    (out / "dense.json").write_text(json.dumps({"model": str(ck.d), "tensors": dense}, indent=0))
    # the same as plain text for the engine: name dtype file offset bytes ndim dims...
    with open(out / "dense.txt", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# model %s\n" % ck.d)
        for name, t in sorted(dense.items()):
            fo.write("%s %s %s %d %d %d %s\n" % (name, t["dtype"], t["file"], t["offset"], t["bytes"], len(t["shape"]),
                                                  " ".join(map(str, t["shape"]))))
    print("dense.json: %d tensors, %.2f GB" % (len(dense), sum(t["bytes"] for t in dense.values()) / 1e9), flush=True)

    # experts.bin, resumable by whole layers
    part = out / "experts.bin.part"
    done = part.stat().st_size // layer_bytes if part.exists() else 0
    if part.exists() and part.stat().st_size != done * layer_bytes:
        with open(part, "r+b") as fh:
            fh.truncate(done * layer_bytes)
    rng = random.Random(0)
    check = {(m, e) for m, e in ((rng.randrange(n_moe), rng.randrange(N_EXPERT)) for _ in range(a.verify))}
    t0 = time.time()
    with open(part, "ab", buffering=64 << 20) as fo:
        for m in range(done, n_moe):
            tl = time.time()
            for e in range(N_EXPERT):
                fo.write(expert_blob(ck, m + DENSE_LEAD, e, verify=(m, e) in check))
                if e % 32 == 31 and private_gb() > a.max_private_gb:
                    print("stopping: %.1f GB committed (limit %.1f)" % (private_gb(), a.max_private_gb), flush=True)
                    return 2
            fo.flush()
            ck.maps.clear()   # drop the views of this layer's shards
            el = time.time() - tl
            print("layer %2d (model %2d): %.1f s, %.2f GB/s, %d of %d; %.1f GB committed"
                  % (m, m + DENSE_LEAD, el, layer_bytes / el / 1e9, m + 1, n_moe, private_gb()), flush=True)
    part.replace(out / "experts.bin")

    tmp = out / "native_experts.txt.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# strata native experts v3: layer gu_type d_type offset blob_bytes (n_expert %d, total %d, "
                 "n_embd %d, n_ff %d, swiglu_limit 10; GLM-5.3-Flash MoE layers %d..%d numbered from 0)\n"
                 % (N_EXPERT, layer_bytes * n_moe, N_EMBD, N_FF, DENSE_LEAD, DENSE_LEAD + n_moe - 1))
        for m in range(n_moe):
            fo.write("%d %d %d %d %d\n" % (m, NVFP4, NVFP4, m * layer_bytes, BLOB))
    tmp.replace(out / "native_experts.txt")
    print("done: %d MoE layers, %.1f GB in %.0f s; %d experts verified value by value"
          % (n_moe, layer_bytes * n_moe / 1e9, time.time() - t0, len(check)), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

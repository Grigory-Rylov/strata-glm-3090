"""tools/glm_cache_sim.py - GLM-5.3-Flash's expert tiers replayed on logged routing (strata-glm --routes).

    python tools/glm_cache_sim.py ROUTES PROMPT_TOKENS [--vram 729] [--ram 7880] [...]

A routes file from a run with a prompt and decode steps: the prompt's chunk records first, then one record per
MoE layer per decode step.  Each policy starts where the engine is after the prompt (the tiers ranked by the
prompt's routing) and is charged, per decode token, for the experts it has to bring from RAM (PCIe) and from
the disk.  The time estimate: RAM hits at --pcie GB/s, disk hits at --disk-ms each, plus --other-ms.

Policies:
  static      the tiers stay as the prompt left them
  rebalance   the engine's: every E tokens the top VRAM pairs by heat move in (at most M), RAM likewise (2M)
  cache       VRAM caches RAM: every RAM hit lands in a VRAM slot (free: it is copied to the GPU anyway),
              evicting the coldest resident by heat (inclusive: RAM keeps its copy); every disk hit lands in a
              RAM slot (free: it is read anyway), evicting RAM's coldest
  lru         the same with least-recently-used eviction
"""
import argparse
import math

import numpy as np

NE, K, NL = 288, 8, 42


def read_routes(path, lead=3):
    blob = np.fromfile(path, dtype=np.int32)
    per, off = {}, 0
    while off + 2 <= len(blob):
        layer, n = int(blob[off]), int(blob[off + 1])
        per.setdefault(layer - lead, []).append(blob[off + 2:off + 2 + n].reshape(-1, K))
        off += 2 + n
    return np.stack([np.concatenate(per[m]) for m in range(NL)])     # [NL, T, K]


class Heat:
    """per pair, a count decayed by half every `half` tokens; weights grow instead of decaying everything"""
    def __init__(self, half):
        self.h = np.zeros(NL * NE)
        self.half, self.t = half, 0.0

    def add(self, pairs, w=1.0):
        np.add.at(self.h, pairs, w * 2.0 ** (self.t / self.half))

    def tick(self, n=1):
        self.t += n
        if self.t > 50 * self.half:      # rescale before the weights overflow
            self.h *= 2.0 ** (-self.t / self.half)
            self.t = 0.0


def run(routes, P, a, policy, w_dec=1.0, every=16, moves=32, half=1024.0):
    T = routes.shape[1]
    pairs = routes + (np.arange(NL) * NE)[:, None, None]             # [NL, T, K] global pair ids
    heat = Heat(half)
    heat.add(pairs[:, :P].ravel())
    heat.tick(P)
    order = np.argsort(-heat.h, kind="stable")
    inclusive = policy in ("cache", "lru")
    vram = np.zeros(NL * NE, bool)
    ram = np.zeros(NL * NE, bool)
    vram[order[:a.vram]] = True
    ram[order[:a.ram] if inclusive else order[a.vram:a.vram + a.ram]] = True
    last = np.zeros(NL * NE)                                          # lru: last use
    last[order] = -np.arange(len(order))
    ram_hits = disk_hits = 0
    for t in range(P, T):
        for m in range(NL):
            for p in pairs[m, t]:
                if vram[p]:
                    pass
                elif ram[p]:
                    ram_hits += 1
                else:
                    disk_hits += 1
                if policy in ("cache", "lru") and not vram[p]:
                    key = last if policy == "lru" else heat.h
                    if not ram[p]:                                   # the disk read lands in RAM
                        cand = np.flatnonzero(ram & ~vram)
                        ram[cand[np.argmin(key[cand])]] = False
                        ram[p] = True
                    cand = np.flatnonzero(vram)                      # the copy to the GPU lands in VRAM
                    vram[cand[np.argmin(key[cand])]] = False
                    vram[p] = True
                last[p] = t
            heat.add(pairs[m, t], w_dec)
        heat.tick()
        if policy == "rebalance" and (t - P + 1) % every == 0:
            order = np.argsort(-heat.h, kind="stable")
            want_v = order[:a.vram]
            inn = [p for p in want_v if not vram[p]][:moves]
            out = [p for p in order[::-1] if vram[p] and p not in set(want_v)][:len(inn)]
            for o, i in zip(out, inn):
                vram[o] = False
                vram[i] = True
                if ram[i]:
                    ram[i] = False                                   # exclusive: its RAM slot is a spare
                ram[o] = True                                        # moved back (approximately)
            want_r = [p for p in order[a.vram:a.vram + a.ram] if not ram[p] and not vram[p]][:2 * moves]
            cold = [p for p in order[::-1] if ram[p]][:len(want_r)]
            for o, i in zip(cold, want_r):
                ram[o] = False
                ram[i] = True
    n = T - P
    rh, dh = ram_hits / n, disk_hits / n
    ms = rh * 14.16e-3 / a.pcie * 1e3 + dh * a.disk_ms + a.other_ms
    return rh, dh, ms


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("routes")
    ap.add_argument("prompt_tokens", type=int)
    ap.add_argument("--vram", type=int, default=729)
    ap.add_argument("--ram", type=int, default=7880)
    ap.add_argument("--pcie", type=float, default=48.0, help="GB/s")
    ap.add_argument("--disk-ms", type=float, default=2.1)
    ap.add_argument("--other-ms", type=float, default=35.0)
    a = ap.parse_args()
    r = read_routes(a.routes)
    print("%d tokens, %d decode; VRAM %d, RAM %d" % (r.shape[1], r.shape[1] - a.prompt_tokens, a.vram, a.ram))
    runs = [("static", {}), ("rebalance", {}), ("rebalance", {"w_dec": 8, "every": 8, "moves": 64}),
            ("rebalance", {"w_dec": 32, "every": 4, "moves": 64}),
            ("cache", {"w_dec": 1}), ("cache", {"w_dec": 8}), ("cache", {"w_dec": 32}), ("cache", {"w_dec": 8, "half": 256}),
            ("lru", {})]
    for pol, kw in runs:
        rh, dh, ms = run(r, a.prompt_tokens, a, pol, **kw)
        print("  %-10s %-38s RAM %5.1f  disk %5.1f a token -> %6.1f ms (%.1f tok/s)" % (
            pol, " ".join("%s=%s" % i for i in kw.items()), rh, dh, ms, 1e3 / ms))


if __name__ == "__main__":
    main()

"""policy/hybrid_sim.py - a static VRAM part excluded from RAM (no write-backs) + an inclusive LRU part.

VRAM = S (the s hottest pairs by the prompt's routing, ties by the boot profile; never evicted, NOT in RAM)
     + an LRU part of V - s slots (a subset of RAM, as the engine's tier today).
RAM  = R slots of pairs outside S, LRU (the victim: not in the VRAM LRU part, not in this layer's selection).
s = 0 is today's inclusive engine. Per decode token: disk reads, RAM->VRAM copies, VRAM hits.
"""
import sys
from collections import OrderedDict
from tier_sim import Trace, NL, K

def run(tr, V, R, s):
    order = tr.order.tolist()
    S = set(order[:s])
    rest = [p for p in order if p not in S]
    ram = OrderedDict((p, None) for p in reversed(rest[:R]))          # LRU first ... MRU last
    vl = OrderedDict((p, None) for p in reversed(rest[:V - s]))        # the LRU part, a subset of RAM
    disk = h2d = hit = 0
    for step in tr.dec_list:
        for sel in step:
            selset = set(sel)
            for p in sel:
                if p in S:
                    hit += 1
                    continue
                if p in vl:
                    hit += 1
                    vl.move_to_end(p)
                    ram.move_to_end(p)
                    continue
                if p in ram:
                    ram.move_to_end(p)
                else:
                    disk += 1
                    for q in ram:                                       # the RAM victim
                        if q not in vl and q not in selset:
                            break
                    del ram[q]
                    ram[p] = None
                h2d += 1
                if len(vl) >= V - s:
                    for q in vl:
                        if q not in selset:
                            break
                    del vl[q]
                vl[p] = None
    D = len(tr.dec_list)
    return disk / D, h2d / D, hit / D

if __name__ == "__main__":
    R = 7070
    for name in ("code", "uk"):
        tr = Trace(name)
        for V in (1455, 1735):
            for s in (0, 600, 900, 1100, 1300, 1500):
                if s >= V - 100:
                    continue
                d, h, v = run(tr, V, R, s)
                print("%-4s V %4d s %4d: disk %5.1f  RAM->VRAM %5.1f  VRAM hits %5.1f  (per token)" % (name, V, s, d, h - d, v), flush=True)

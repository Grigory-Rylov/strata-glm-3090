"""policy/hybrid_resel.py - hybrid_sim with the static part re-chosen every P decode tokens.

score(pair) = decode uses decayed by 2^(-age/H) + w * prompt count / prompt tokens * P_scale. At each re-selection the
top-s pairs become static: one already in the VRAM LRU part just changes role (its RAM copy is freed); one elsewhere
is copied in (counted as H2D) and takes the place of the LRU part's least recent; a pair leaving the static part
joins the LRU part and needs its RAM copy back (counted as a D2H write-back, the idle direction).
"""
import math
from collections import OrderedDict
import numpy as np
from tier_sim import Trace, NP

def run(tr, V, R, s, P=0, H=256.0, w=1.0, maxm=10**9, hyst=1.0):
    order = tr.order.tolist()
    S = set(order[:s])
    rest = [p for p in order if p not in S]
    ram = OrderedDict((p, None) for p in reversed(rest[:R]))
    vl = OrderedDict((p, None) for p in reversed(rest[:V - s]))
    use = np.zeros(NP)
    base = w * tr.cnt / max(1, tr.P)
    disk = h2d = hit = d2h = 0
    for t, step in enumerate(tr.dec_list):
        if P and t > 0 and t % P == 0:
            score = use + base
            top = set(np.argsort(-score)[:s].tolist())
            ent = sorted(top - S, key=lambda q: -score[q])
            lea = sorted(S - top, key=lambda q: score[q])
            n = 0
            while n < min(len(ent), len(lea), maxm) and score[ent[n]] > hyst * score[lea[n]]:
                n += 1
            top = (S - set(lea[:n])) | set(ent[:n])
            for p in list(S - top):            # leaving: back to the LRU part (MRU), its RAM copy written back
                S.discard(p)
                vl[p] = None
                d2h += 1
                if p not in ram and len(ram) >= R:
                    for q in ram:
                        if q not in vl and q not in S:
                            break
                    del ram[q]
                ram[p] = None
            for p in top - S:                  # entering
                if p in vl:
                    del vl[p]
                else:
                    h2d += 1
                if p in ram:
                    del ram[p]
                S.add(p)
            while len(vl) > V - s:             # the LRU part back to its size
                q = next(iter(vl))
                del vl[q]
        for sel in step:
            selset = set(sel)
            for p in sel:
                use *= 1.0                     # (decay applied per token below)
                if p in S:
                    hit += 1
                elif p in vl:
                    hit += 1
                    vl.move_to_end(p)
                    ram.move_to_end(p)
                else:
                    if p in ram:
                        ram.move_to_end(p)
                    else:
                        disk += 1
                        if len(ram) >= R:
                            for q in ram:
                                if q not in vl and q not in selset and q not in S:
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
                use[p] += 1.0
        use *= math.pow(2.0, -1.0 / H)
    D = len(tr.dec_list)
    return disk / D, (h2d - disk) / D, hit / D, d2h / D

if __name__ == "__main__":
    for name in ("code", "uk"):
        tr = Trace(name)
        for P, maxm, hyst in ((0, 0, 1), (16, 10**9, 1), (16, 32, 1.5), (16, 16, 1.5), (8, 16, 1.5), (8, 8, 2.0), (4, 8, 1.5)):
            d, r, v, b = run(tr, 1543, 7070, 1080, P, 128, 1.0, maxm, hyst)
            print("%-4s P %2d moves %4s hyst %.1f: disk %5.1f  RAM->VRAM %5.1f  VRAM hits %5.1f  D2H %4.1f  PCIe in %5.1f"
                  % (name, P, maxm if maxm < 10**8 else "all", hyst, d, r, v, b, d + r), flush=True)

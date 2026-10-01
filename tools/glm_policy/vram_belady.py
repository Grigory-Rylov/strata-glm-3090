"""policy/vram_belady.py - VRAM hits per decode token at a fixed capacity: LRU, the static+LRU hybrid, Belady MIN."""
import heapq
from collections import OrderedDict
from tier_sim import Trace

def lru(tr, V):
    order = tr.order.tolist()
    v = OrderedDict((p, None) for p in reversed(order[:V]))
    hit = 0
    for step in tr.dec_list:
        for sel in step:
            for p in sel:
                if p in v: hit += 1; v.move_to_end(p)
                else:
                    if len(v) >= V:
                        for q in v:
                            if q not in sel: break
                        del v[q]
                    v[p] = None
    return hit / len(tr.dec_list)

def belady(tr, V):
    order = tr.order.tolist()
    pos = [0] * len(tr.clocks_of)
    INF = float("inf")
    def next_use(p, now):   # the first access strictly after `now`
        c = tr.clocks_of[p]
        i = pos[p]
        while i < len(c) and c[i] <= now: i += 1
        pos[p] = i
        return c[i] if i < len(c) else INF
    res = set(order[:V])
    heap = [(-next_use(p, -1), p) for p in res]
    heapq.heapify(heap)
    hit = 0
    clock = 0
    for step in tr.dec_list:
        for sel in step:
            for p in sel:
                if p in res:
                    hit += 1
                else:
                    held = []
                    while True:
                        nu, q = heapq.heappop(heap)
                        if q not in res: continue
                        cur = next_use(q, clock)
                        if -nu != cur: heapq.heappush(heap, (-cur, q)); continue
                        if q in sel: held.append((nu, q)); continue
                        break
                    for h in held: heapq.heappush(heap, h)
                    res.discard(q)
                    res.add(p)
                heapq.heappush(heap, (-next_use(p, clock), p))
            clock += 1
    return hit / len(tr.dec_list)

for name in ("code", "uk"):
    tr = Trace(name)
    print(name, "LRU %.1f  Belady %.1f (VRAM hits a token of 336, V=1543)" % (lru(tr, 1543), belady(tr, 1543)), flush=True)

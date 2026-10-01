"""policy/tier_sim.py - GLM-5.3-Flash expert tiers replayed on logged routing: VRAM (LRU, inclusive) over RAM over disk.

The model (strata-glm --policy lru, src/glm/glm_main.cpp, decode path):
  * 12096 pairs (42 MoE layers x 288 experts), top-8 per layer per token.
  * VRAM: 1455 slots. Every expert a layer uses ends up in VRAM: a VRAM hit touches it, a RAM hit or a disk read is
    copied into the VRAM slot of the least recently used resident outside this layer's selection.
  * RAM: 7070 slots, inclusive (VRAM is always a subset of RAM). A disk miss is read into the slot of a RAM member
    that is not in VRAM and not in this layer's selection - the RAM POLICY picks it (engine: rlru tail-walk).
  * Before decode, the rebalance: VRAM = top 1455 by the prompt's routing counts, RAM = top 7070 (ties: the boot
    profile's rank, as the engine's `prior`); the LRU recency is that ranking (hottest = most recent).
  * Per layer, in the engine's order: VRAM hits touch; RAM hits leave the victim pool; disk victims are chosen;
    RAM hits and disk reads are copied into VRAM, each evicting VRAM's LRU (which then becomes a RAM candidate).
  * The prefetcher is NOT modelled in the main runs (pf=None). `pf` adds a stochastic stand-in (see Prefetch).

    python tier_sim.py                 # the classical policies, both prompts -> results_ram.json / results_ram.txt
Companions (same directory): final.py (the curated table -> results.txt), sweep_est.py, plateau.py, decay_check.py,
rank_prior_check.py, explore2.py, check_vs_old.py (LRU against tools/glm_cache_sim.py).
"""
import heapq
import json
import math
import os
import struct
import sys
import time
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))   # tools/glm_cache_sim.py
from glm_cache_sim import read_routes  # noqa: E402

NE, K, NL = 288, 8, 42
NP = NL * NE
V_SLOTS, R_SLOTS = 1455, 7070
DATA = os.environ.get("GLM_DATA", "data")   # the routing traces (strata-glm --routes) and profile_boot/
OUT = os.path.join(DATA, "policy")
PROFILE = os.environ.get("GLM_PROFILE", "expert-profile-boot.bin")
TRACES = {"code": ("routes_chat_code.bin", 849), "uk": ("routes_chat_uk.bin", 1394)}
INF = float("inf")
NINF = -INF


def read_profile_rank(path):
    b = open(path, "rb").read()
    assert b[:4] == b"STRP"
    _, nl, ne, _, n = struct.unpack("<5I", b[4:24])
    assert (nl, ne) == (NL, NE)
    raw = np.frombuffer(b[24:24 + 4 * n], dtype="<u2").reshape(-1, 2).astype(np.int64)
    rank = np.full(NP, NP, np.int64)
    rank[raw[:, 0] * NE + raw[:, 1]] = np.arange(n)
    return rank


class Trace:
    def __init__(self, name, extra_prior=None, beta=0.0):
        f, P = TRACES[name]
        r = read_routes(os.path.join(DATA, f))
        self.name, self.P = name, P
        pairs = (r + (np.arange(NL) * NE)[:, None, None]).astype(np.int64)
        self.prompt = pairs[:, :P]                                            # [NL, P, K]
        self.dec = np.ascontiguousarray(pairs[:, P:].transpose(1, 0, 2))      # [D, NL, K]
        self.D = self.dec.shape[0]
        self.cnt = np.bincount(self.prompt.ravel(), minlength=NP).astype(np.float64)
        self.prior = read_profile_rank(PROFILE)
        heat = self.cnt if extra_prior is None else self.cnt / self.cnt.sum() + beta * extra_prior / extra_prior.sum()
        self.order = np.lexsort((self.prior, -heat))                          # hottest first, ties by the profile
        # prompt history: (pair, token) for every prompt access
        tok = np.broadcast_to(np.arange(P)[None, :, None], self.prompt.shape)
        self.p_pairs = self.prompt.ravel()
        self.p_tok = tok.ravel().astype(np.float64)
        # decode access clocks per pair (clock = step * NL + layer), for Belady
        flat = self.dec.reshape(-1)
        clocks = np.arange(flat.size) // K
        o = np.argsort(flat, kind="stable")
        self.clocks_of = [[] for _ in range(NP)]
        for p, c in zip(flat[o].tolist(), clocks[o].tolist()):
            self.clocks_of[p].append(c)
        self.dec_list = self.dec.tolist()
        # generic priors, per-token use probability of each pair (sums to 8 per layer):
        #   boot  - the prefill routing the engine's boot profile was ranked from (profile_boot/long_00.routes)
        #   other - the OTHER conversation's decode (a leave-one-out stand-in for a shipped decode profile)
        rb = read_routes(os.path.join(DATA, "profile_boot", "long_00.routes"))
        self.g_boot = np.bincount((rb + (np.arange(NL) * NE)[:, None, None]).ravel(), minlength=NP) / rb.shape[1]
        of, oP = [v for k, v in TRACES.items() if k != name][0]
        ro = read_routes(os.path.join(DATA, of))[:, oP:]
        self.g_other = np.bincount((ro + (np.arange(NL) * NE)[:, None, None]).ravel(), minlength=NP) / ro.shape[1]


def addlog(a, b):   # log2(2^a + 2^b)
    if a < b:
        a, b = b, a
    if b == NINF:
        return a
    return a + math.log2(1.0 + 2.0 ** (b - a))


# ---------------------------------------------------------------- RAM policies
class Policy:
    """interface: init, touch (a resident used: VRAM or RAM hit), victim (a disk miss needs a slot: a RAM member
    outside VRAM; the caller has marked this layer's selection as in VRAM), insert (the missed pair lands),
    vram_leave (VRAM evicted it: it becomes a candidate)"""
    label = "?"

    def init(self, tr, in_ram, in_vram):
        self.tr, self.in_ram, self.in_vram = tr, in_ram, in_vram

    def touch(self, p, now, clock):
        pass

    def insert(self, p, now, clock):
        pass

    def vram_leave(self, q, now):
        pass

    def victim(self, now, x):
        raise NotImplementedError

    def cand(self, q):
        return self.in_ram[q] and not self.in_vram[q]


class Prio(Policy):
    """evict the candidate with the smallest key. A RAM-only member is never touched (any use moves it to VRAM), so
    its key is fixed from the moment VRAM lets it go: a lazy min-heap with a version per pair is exact."""

    def init(self, tr, in_ram, in_vram):
        super().init(tr, in_ram, in_vram)
        self.ver = [0] * NP
        self.setup(tr)
        self.heap = [(self.key(q), 0, q) for q in range(NP) if in_ram[q] and not in_vram[q]]
        heapq.heapify(self.heap)

    def setup(self, tr):
        self.last = [NINF] * NP
        for i, p in enumerate(tr.order.tolist()):   # the rebalance's recency: hottest = most recent, all at the
            self.last[p] = tr.P - 1e-6 * (i + 1)     # decode start (same time base as `now`: tokens)

    def note(self, p, now, clock):
        self.last[p] = now

    def touch(self, p, now, clock):
        self.note(p, now, clock)

    def insert(self, p, now, clock, pf=False):
        if pf:   # a prefetch read lands: recent (the engine touches it), but not a use
            self.last[p] = now
        else:
            self.note(p, now, clock)

    def vram_leave(self, q, now):
        self.ver[q] += 1
        heapq.heappush(self.heap, (self.key(q), self.ver[q], q))

    def victim(self, now, x):
        h, ver = self.heap, self.ver
        while h:
            _, v, q = heapq.heappop(h)
            if v == ver[q] and self.in_ram[q] and not self.in_vram[q]:
                ver[q] += 1
                return q
        raise RuntimeError("no RAM victim")


class LRU(Prio):
    label = "LRU (engine today)"

    def key(self, q):
        return self.last[q]


class LFUDecay(Prio):
    """score = w * prompt history + decode uses, each use weighted 2^(-(now - t)/half) (LRFU / CRF).  mode 'flat':
    the prompt counts are dated at the decode start; 'timed': each prompt use at its own token.  half=inf: LFU."""

    def __init__(self, half=INF, w=1.0, mode="flat"):
        self.half, self.w, self.mode = half, w, mode
        self.label = "LFU-decay h=%s w=%g %s" % (half, w, mode)

    def setup(self, tr):
        super().setup(tr)
        h, P = self.half, tr.P
        if self.w <= 0:
            base = np.zeros(NP)
        elif self.mode == "flat" or h == INF:
            base = self.w * tr.cnt
        else:
            base = self.w * np.bincount(tr.p_pairs, weights=2.0 ** ((tr.p_tok - P) / h), minlength=NP)
        if h == INF:
            self.s = base.tolist()
        else:
            with np.errstate(divide="ignore"):
                self.s = np.log2(base).tolist()                                # -inf for never seen

    def note(self, p, now, clock):
        super().note(p, now, clock)
        if self.half == INF:
            self.s[p] += 1.0
        else:
            self.s[p] = addlog(self.s[p], (now - self.tr.P) / self.half)

    def key(self, q):
        return (self.s[q], self.last[q])


class LRUK(Prio):
    """LRU-K: evict the oldest K-th most recent use (fewer than K uses: first, by last use); history includes the
    prompt's uses at their tokens"""

    def __init__(self, k=2, prompt=True):
        self.k, self.prompt = k, prompt
        self.label = "LRU-%d%s" % (k, "" if prompt else " (no prompt hist)")

    def setup(self, tr):
        super().setup(tr)
        self.hist = [[] for _ in range(NP)]
        if self.prompt:
            o = np.lexsort((tr.p_tok, tr.p_pairs))
            for p, t in zip(tr.p_pairs[o].tolist(), tr.p_tok[o].tolist()):
                hh = self.hist[p]
                if not hh or hh[-1] != t:
                    hh.append(t)
                    if len(hh) > self.k:
                        del hh[0]

    def note(self, p, now, clock):
        super().note(p, now, clock)
        hh = self.hist[p]
        hh.append(now)
        if len(hh) > self.k:
            del hh[0]

    def key(self, q):
        hh = self.hist[q]
        return (hh[0] if len(hh) >= self.k else NINF, self.last[q])


class Boost(Prio):
    """frequency-boosted LRU: key = last use (tokens) + tn*log2(1 + decode uses) + tc*log2(1 + prompt count), i.e.
    a pair counts as used more recently by an amount that grows with how often it was used (GreedyDual-like)"""

    def __init__(self, tn=8.0, tc=8.0, init="rank"):
        self.tn, self.tc, self.init_mode = tn, tc, init
        self.label = "boosted LRU tn=%g tc=%g%s" % (tn, tc, "" if init == "rank" else " init=" + init)

    def setup(self, tr):
        super().setup(tr)
        self.n = [0] * NP
        self.bc = (self.tc * np.log2(1.0 + tr.cnt)).tolist()
        if self.init_mode == "prompt":   # last use = the last prompt token that routed to it
            lt = np.full(NP, 0.0)
            np.maximum.at(lt, tr.p_pairs, tr.p_tok)
            self.last = lt.tolist()

    def note(self, p, now, clock):
        super().note(p, now, clock)
        self.n[p] += 1

    def key(self, q):
        return self.last[q] + self.tn * math.log2(1.0 + self.n[q]) + self.bc[q]


class Est(Prio):
    """the use-rate estimate n + Ap*pi_prompt + Ag*pi_generic (decode uses, plus Ap and Ag tokens' worth of the
    prompt's and a generic profile's per-token rates: a Bayesian rate with those priors; the common denominator
    drops out of the ranking).  tau=inf: evict the lowest estimate, ties by LRU; finite tau: key = last +
    tau*log2(estimate) (a doubling of the estimate is worth tau tokens of recency).  half: decode uses decay."""

    def __init__(self, ap=0.0, ag=0.0, tau=INF, half=INF, g="boot", eps=0.01):
        self.ap, self.ag, self.tau, self.half, self.g, self.eps = ap, ag, tau, half, g, eps
        self.label = "rate-est Ap=%g Ag=%g(%s) tau=%s%s" % (ap, ag, g, tau, "" if half == INF else " h=%g" % half)

    def setup(self, tr):
        super().setup(tr)
        g = tr.g_boot if self.g == "boot" else tr.g_other
        self.base = (self.ap * tr.cnt / tr.P + self.ag * g + self.eps).tolist()
        self.n = [0.0] * NP

    def note(self, p, now, clock):
        super().note(p, now, clock)
        self.n[p] += 1.0 if self.half == INF else 2.0 ** ((now - self.tr.P) / self.half)

    def key(self, q):
        e = self.n[q] + self.base[q]
        if self.tau == INF:
            return (e, self.last[q])
        return self.last[q] + self.tau * math.log2(e)


class Protected(Prio):
    """LRU, but the top-N pairs by the prompt's counts are evicted only when nothing else can be"""

    def __init__(self, n=4000):
        self.n = n
        self.label = "prompt-protected LRU N=%d" % n

    def setup(self, tr):
        super().setup(tr)
        self.prot = [0] * NP
        for p in tr.order[:self.n].tolist():
            self.prot[p] = 1

    def key(self, q):
        return (self.prot[q], self.last[q])


class Belady(Prio):
    """MIN: evict the candidate used furthest in the future (the ceiling for any RAM policy, VRAM fixed as LRU)"""
    label = "Belady MIN (ceiling)"

    def setup(self, tr):
        super().setup(tr)
        self.ptr = [0] * NP
        self.co = tr.clocks_of

    def nxt(self, q):
        c = self.co[q]
        i = self.ptr[q]
        return c[i] if i < len(c) else INF

    def note(self, p, now, clock):
        super().note(p, now, clock)
        self.ptr[p] += 1

    def key(self, q):
        return -self.nxt(q)


class LayerQuota(Policy):
    """per-layer budgets: a miss in layer m evicts from layer m when m is at its quota, else from the over-quota
    layer with the weakest candidate; inside a layer the order is `inner`'s key (LRU or LFU-decay)"""

    def __init__(self, inner, quota="prompt"):
        self.inner, self.quota = inner, quota
        self.label = "layer-quota(%s) + %s" % (quota, inner.label)

    def init(self, tr, in_ram, in_vram):
        super().init(tr, in_ram, in_vram)
        inn = self.inner
        inn.in_ram, inn.in_vram, inn.tr = in_ram, in_vram, tr
        inn.setup(tr)
        self.count = [0] * NL
        for q in range(NP):
            if in_ram[q]:
                self.count[q // NE] += 1
        if self.quota == "prompt":
            self.q = list(self.count)
        else:
            base = R_SLOTS // NL
            self.q = [base + (1 if m < R_SLOTS - base * NL else 0) for m in range(NL)]
        self.ver = [0] * NP
        self.heaps = [[] for _ in range(NL)]
        for q in range(NP):
            if in_ram[q] and not in_vram[q]:
                self.heaps[q // NE].append((inn.key(q), 0, q))
        for h in self.heaps:
            heapq.heapify(h)

    def touch(self, p, now, clock):
        self.inner.note(p, now, clock)

    def insert(self, p, now, clock):
        self.inner.note(p, now, clock)
        self.count[p // NE] += 1

    def vram_leave(self, q, now):
        self.ver[q] += 1
        heapq.heappush(self.heaps[q // NE], (self.inner.key(q), self.ver[q], q))

    def top(self, m):
        h = self.heaps[m]
        while h:
            _, v, q = h[0]
            if v == self.ver[q] and self.in_ram[q] and not self.in_vram[q]:
                return h[0]
            heapq.heappop(h)
        return None

    def victim(self, now, x):
        m = x // NE
        if self.count[m] >= self.q[m] and self.top(m) is not None:
            src = m
        else:
            best, src = None, -1
            for l in range(NL):
                if l != m and self.count[l] > self.q[l]:
                    t = self.top(l)
                    if t is not None and (best is None or t < best):
                        best, src = t, l
            if src < 0:   # nobody over quota with a candidate: the weakest anywhere
                for l in range(NL):
                    t = self.top(l)
                    if t is not None and (best is None or t < best):
                        best, src = t, l
        _, _, q = heapq.heappop(self.heaps[src])
        self.ver[q] += 1
        self.count[src] -= 1
        return q


def first_cand(od, in_vram):
    for q in od:
        if not in_vram[q]:
            return q
    return None


class ARC(Policy):
    """ARC (Megiddo & Modha) over all RAM members, c = 7070; a replacement takes the least recent T1/T2 entry
    outside VRAM. Starts with the rebalance's set in T2 (hottest most recent), p = 0."""

    def __init__(self):
        self.label = "ARC"

    def init(self, tr, in_ram, in_vram):
        super().init(tr, in_ram, in_vram)
        self.c = R_SLOTS
        self.p = 0.0
        self.T1, self.T2, self.B1, self.B2 = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict()
        for q in reversed(tr.order[:R_SLOTS].tolist()):
            self.T2[q] = None

    def touch(self, x, now, clock):
        if x in self.T1:
            del self.T1[x]
            self.T2[x] = None
        else:
            self.T2.move_to_end(x)

    def _replace(self, x):
        T1, T2 = self.T1, self.T2
        use_t1 = len(T1) >= 1 and ((x in self.B2 and len(T1) == self.p) or len(T1) > self.p)
        q = first_cand(T1 if use_t1 else T2, self.in_vram)
        if q is None:
            use_t1 = not use_t1
            q = first_cand(T1 if use_t1 else T2, self.in_vram)
        if use_t1:
            del T1[q]
            self.B1[q] = None
        else:
            del T2[q]
            self.B2[q] = None
        return q

    def victim(self, now, x):
        T1, T2, B1, B2, c = self.T1, self.T2, self.B1, self.B2, self.c
        if x in B1:
            self.p = min(c, self.p + max(len(B2) / len(B1), 1.0))
            return self._replace(x)
        if x in B2:
            self.p = max(0.0, self.p - max(len(B1) / len(B2), 1.0))
            return self._replace(x)
        if len(T1) + len(B1) >= c:
            if len(T1) < c:
                B1.popitem(last=False)
                return self._replace(x)
            q = first_cand(T1, self.in_vram)
            del T1[q]
            return q
        if len(T1) + len(T2) + len(B1) + len(B2) >= 2 * c and B2:
            B2.popitem(last=False)
        return self._replace(x)

    def insert(self, x, now, clock):
        if x in self.B1:
            del self.B1[x]
            self.T2[x] = None
        elif x in self.B2:
            del self.B2[x]
            self.T2[x] = None
        else:
            self.T1[x] = None


class TwoQ(Policy):
    """2Q (Johnson & Shasha, full version): A1in FIFO of new pairs (Kin), Am LRU, A1out ghosts (Kout).
    promote=True: a use while in A1in moves the pair to Am (simplified 2Q)."""

    def __init__(self, kin=0.25, kout=0.5, promote=False):
        self.kin, self.kout, self.promote = kin, kout, promote
        self.label = "2Q Kin=%.2fc Kout=%.2fc%s" % (kin, kout, " promote" if promote else "")

    def init(self, tr, in_ram, in_vram):
        super().init(tr, in_ram, in_vram)
        self.Kin, self.Kout = int(self.kin * R_SLOTS), int(self.kout * R_SLOTS)
        self.Am, self.A1in, self.A1out = OrderedDict(), OrderedDict(), OrderedDict()
        for q in reversed(tr.order[:R_SLOTS].tolist()):
            self.Am[q] = None

    def touch(self, x, now, clock):
        if x in self.Am:
            self.Am.move_to_end(x)
        elif self.promote and x in self.A1in:
            del self.A1in[x]
            self.Am[x] = None

    def victim(self, now, x):
        src = self.A1in if len(self.A1in) > self.Kin else self.Am
        q = first_cand(src, self.in_vram)
        if q is None:
            src = self.Am if src is self.A1in else self.A1in
            q = first_cand(src, self.in_vram)
        del src[q]
        if src is self.A1in:
            self.A1out[q] = None
            if len(self.A1out) > self.Kout:
                self.A1out.popitem(last=False)
        return q

    def insert(self, x, now, clock):
        if x in self.A1out:
            del self.A1out[x]
            self.Am[x] = None
        else:
            self.A1in[x] = None


class LIRS(Policy):
    """LIRS (Jiang & Zhang): LIR set c - Lhirs, resident-HIR queue Q; the victim is Q's oldest entry outside VRAM
    (none: the bottom-most LIR outside VRAM, and the newcomer takes its LIR place)"""
    LIR, HIR, GHOST = 1, 2, 3

    def __init__(self, hir=0.1):
        self.hir = hir
        self.label = "LIRS Lhirs=%.2fc" % hir

    def init(self, tr, in_ram, in_vram):
        super().init(tr, in_ram, in_vram)
        Lh = max(1, int(self.hir * R_SLOTS))
        Ll = R_SLOTS - Lh
        self.st = [0] * NP
        self.S, self.Q = OrderedDict(), OrderedDict()
        order = tr.order.tolist()
        for q in reversed(order[:Ll]):
            self.S[q] = None
            self.st[q] = self.LIR
        for q in reversed(order[Ll:R_SLOTS]):
            self.Q[q] = None
            self.st[q] = self.HIR
        self.lir_debt = False

    def _prune(self):
        S, st = self.S, self.st
        while S:
            q = next(iter(S))
            if st[q] == self.LIR:
                break
            del S[q]
            if st[q] == self.GHOST:
                st[q] = 0

    def _demote_bottom(self):
        y = next(iter(self.S))
        del self.S[y]
        self.st[y] = self.HIR
        self.Q[y] = None
        self._prune()

    def touch(self, x, now, clock):
        S, st = self.S, self.st
        if st[x] == self.LIR:
            bottom = next(iter(S)) == x
            S.move_to_end(x)
            if bottom:
                self._prune()
        else:   # resident HIR
            if x in S:
                S.move_to_end(x)
                st[x] = self.LIR
                del self.Q[x]
                self._demote_bottom()
            else:
                S[x] = None
                self.Q.move_to_end(x)

    def victim(self, now, x):
        q = first_cand(self.Q, self.in_vram)
        if q is not None:
            del self.Q[q]
            self.st[q] = self.GHOST if q in self.S else 0
            return q
        for q in self.S:   # bottom-most LIR outside VRAM
            if self.st[q] == self.LIR and not self.in_vram[q]:
                break
        del self.S[q]
        self.st[q] = 0
        self.lir_debt = True
        self._prune()
        return q

    def insert(self, x, now, clock):
        S, st = self.S, self.st
        if self.lir_debt:   # the evicted LIR's place
            self.lir_debt = False
            if x in S:
                S.move_to_end(x)
            else:
                S[x] = None
            st[x] = self.LIR
            return
        if st[x] == self.GHOST:
            S.move_to_end(x)
            st[x] = self.LIR
            self._demote_bottom()
        else:
            S[x] = None
            st[x] = self.HIR
            self.Q[x] = None


# ---------------------------------------------------------------- VRAM policies
class VLRU:
    label = "LRU"

    def __init__(self, tr):
        self.od = OrderedDict((q, None) for q in reversed(tr.order[:V_SLOTS].tolist()))

    def touch(self, p, now):
        self.od.move_to_end(p)

    def insert(self, p, now, S):
        for q in self.od:
            if q not in S:
                break
        del self.od[q]
        self.od[p] = None
        return q


class VLFU:
    """VRAM by LFU-decay (same score as the RAM policy's); a newcomer starts at its own score"""

    def __init__(self, tr, half=INF, w=1.0, scorer=None):
        self.label = "LFU-decay h=%s w=%g" % (half, w) if scorer is None else "VRAM " + scorer.label
        self.sc = LFUDecay(half, w) if scorer is None else scorer
        self.sc.tr = tr
        self.sc.setup(tr)
        self.ver = [0] * NP
        self.res = bytearray(NP)
        self.heap = []
        for q in tr.order[:V_SLOTS].tolist():
            self.res[q] = 1
            self.heap.append((self.sc.key(q), 0, q))
        heapq.heapify(self.heap)

    def touch(self, p, now):
        self.sc.note(p, now, 0)
        self.ver[p] += 1
        heapq.heappush(self.heap, (self.sc.key(p), self.ver[p], p))

    def insert(self, p, now, S):
        held = []
        while True:
            e = heapq.heappop(self.heap)
            q = e[2]
            if e[1] != self.ver[q] or not self.res[q]:
                continue
            if q in S:
                held.append(e)
                continue
            break
        for e in held:
            heapq.heappush(self.heap, e)
        self.res[q] = 0
        self.ver[q] += 1
        self.res[p] = 1
        self.touch(p, now)
        return q


# ---------------------------------------------------------------- the replay
def simulate(tr, rpol, vpol=None, pf=None, seed=1):
    """pf: None, or (acc list per predicted rank, wrong-pick weights 'prompt'|'uniform'): before each layer the
    stand-in prefetcher reads its predictions that are not in RAM into RAM (counted as prefetch reads; a correct one
    turns that layer's disk miss into a RAM hit)"""
    in_ram, in_vram = bytearray(NP), bytearray(NP)
    order = tr.order.tolist()
    for q in order[:R_SLOTS]:
        in_ram[q] = 1
    for q in order[:V_SLOTS]:
        in_vram[q] = 1
    vp = VLRU(tr) if vpol is None else vpol(tr)
    rpol.init(tr, in_ram, in_vram)
    D, P = tr.D, tr.P
    vh, rh, dm, pr = [0] * D, [0] * D, [0] * D, [0] * D
    rng = np.random.default_rng(seed)
    if pf is not None:
        acc = pf[0]
        wl = tr.cnt.reshape(NL, NE) + 1.0
    for d in range(D):
        row = tr.dec_list[d]
        for m in range(NL):
            S = row[m]
            clock = d * NL + m
            now = P + d + m / NL
            if pf is not None:   # the previous layer's prediction of this one, read ahead (VRAM untouched)
                Sset = set(S)
                picks, used = [], set()
                true_ids = list(S)
                rng.shuffle(true_ids)
                for a in acc:
                    if rng.random() < a and true_ids:
                        q = true_ids.pop()
                    else:
                        w = wl[m].copy()
                        for e in S:
                            w[e - m * NE] = 0
                        q = m * NE + int(rng.choice(NE, p=w / w.sum()))
                    if q not in used:
                        used.add(q)
                        picks.append(q)
                for q in picks:   # issued while layer m-1 ran: its selection is in VRAM already
                    if in_ram[q] or in_vram[q]:
                        continue
                    v = rpol.victim(now, q)
                    in_ram[v] = 0
                    in_ram[q] = 1
                    if isinstance(rpol, Prio):
                        rpol.insert(q, now, clock, pf=True)
                    else:
                        rpol.insert(q, now, clock)
                    rpol.vram_leave(q, now)   # RAM-only: a candidate at once
                    pr[d] += 1
            hr, ms = [], []
            for p in S:
                if in_vram[p]:
                    vh[d] += 1
                    vp.touch(p, now)
                    rpol.touch(p, now, clock)
                elif in_ram[p]:
                    hr.append(p)
                else:
                    ms.append(p)
            for p in hr:              # leaving the victim pool (the engine's in_sel)
                in_vram[p] = 1
            for p in ms:
                q = rpol.victim(now, p)
                in_ram[q] = 0
                in_ram[p] = 1
                in_vram[p] = 1
                rpol.insert(p, now, clock)
            for p in hr:
                rpol.touch(p, now, clock)
            for p in hr + ms:
                q = vp.insert(p, now, S)
                in_vram[q] = 0
                rpol.vram_leave(q, now)
            rh[d] += len(hr)
            dm[d] += len(ms)
    assert sum(in_ram) == R_SLOTS and sum(in_vram) == V_SLOTS
    assert all(in_ram[q] for q in range(NP) if in_vram[q])
    return np.array(vh), np.array(rh), np.array(dm), np.array(pr)


def simulate_excl(tr, rpol, bypass=True):
    """--ram-exclusive: VRAM and RAM disjoint (8525 distinct experts cached).  A RAM hit moves to VRAM and frees its
    RAM slot; a disk read goes to VRAM (through a transient RAM slot); every VRAM eviction is offered to RAM (a D2H
    write-back) and RAM, if full, drops its policy victim - with bypass the evictee itself may lose (no write-back).
    Only Prio policies (their keys work unchanged).  Returns vh, rh, dm, write-backs per token."""
    in_ram, in_vram = bytearray(NP), bytearray(NP)
    order = tr.order.tolist()
    for q in order[:V_SLOTS]:
        in_vram[q] = 1
    for q in order[V_SLOTS:V_SLOTS + R_SLOTS]:
        in_ram[q] = 1
    nram = R_SLOTS
    vp = VLRU(tr)
    rpol.init(tr, in_ram, in_vram)
    D, P = tr.D, tr.P
    vh, rh, dm, wb = [0] * D, [0] * D, [0] * D, [0] * D
    for d in range(D):
        row = tr.dec_list[d]
        for m in range(NL):
            S = row[m]
            clock = d * NL + m
            now = P + d + m / NL
            hr, ms = [], []
            for p in S:
                if in_vram[p]:
                    vh[d] += 1
                    vp.touch(p, now)
                    rpol.touch(p, now, clock)
                elif in_ram[p]:
                    hr.append(p)
                else:
                    ms.append(p)
            for p in hr:
                in_ram[p] = 0
                nram -= 1
                in_vram[p] = 1
                rpol.touch(p, now, clock)
            for p in ms:
                in_vram[p] = 1
                rpol.insert(p, now, clock)
            for p in hr + ms:
                q = vp.insert(p, now, S)
                in_vram[q] = 0
                in_ram[q] = 1
                nram += 1
                rpol.vram_leave(q, now)
                if nram > R_SLOTS:
                    if not bypass:
                        in_vram[q] = 1          # the evictee is written back whatever its key
                    v = rpol.victim(now, -1)
                    if not bypass:
                        in_vram[q] = 0
                        rpol.vram_leave(q, now)
                    in_ram[v] = 0
                    nram -= 1
                    wb[d] += v != q
                else:
                    wb[d] += 1
            rh[d] += len(hr)
            dm[d] += len(ms)
    assert nram == sum(in_ram) <= R_SLOTS and sum(in_vram) == V_SLOTS
    assert not any(in_ram[q] and in_vram[q] for q in range(NP))
    return np.array(vh), np.array(rh), np.array(dm), np.array(wb)


# ---------------------------------------------------------------- the sweep
def specs():
    out = [("LRU", {}), ("Belady", {})]
    for h in [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, INF]:
        for w in [0.0, 0.25, 1.0]:
            out.append(("LFUDecay", {"half": h, "w": w, "mode": "flat"}))
        if h != INF:
            out.append(("LFUDecay", {"half": h, "w": 1.0, "mode": "timed"}))
    for h in [32, 128, 512, INF]:
        for w in [0.05, 0.1, 4.0]:
            out.append(("LFUDecay", {"half": h, "w": w, "mode": "flat"}))
    for k in [2, 3]:
        out.append(("LRUK", {"k": k}))
        out.append(("LRUK", {"k": k, "prompt": False}))
    for n in [1500, 3000, 4500, 6000]:
        out.append(("Protected", {"n": n}))
    out.append(("ARC", {}))
    for kin in [0.1, 0.25, 0.5]:
        out.append(("TwoQ", {"kin": kin, "kout": 0.5}))
    out.append(("TwoQ", {"kin": 0.25, "kout": 1.0, "promote": True}))
    for hir in [0.01, 0.1, 0.3, 0.5]:
        out.append(("LIRS", {"hir": hir}))
    for q in ["prompt", "uniform"]:
        out.append(("LayerQuota", {"inner": ("LRU", {}), "quota": q}))
        out.append(("LayerQuota", {"inner": ("LFUDecay", {"half": 128, "w": 0.25}), "quota": q}))
        out.append(("LayerQuota", {"inner": ("LFUDecay", {"half": INF, "w": 0.25}), "quota": q}))
    return out


def make(spec):
    name, kw = spec
    kw = dict(kw)
    if "inner" in kw:
        kw["inner"] = make(kw["inner"])
    return globals()[name](**kw)


_TR = {}


def trace(name):
    if name not in _TR:
        _TR[name] = Trace(name)
    return _TR[name]


def job(args):
    tname, spec, vspec, pf = args
    tr = trace(tname)
    pol = make(spec)
    vpol = None
    if vspec is not None:
        vpol = lambda t: VLFU(t, **vspec)  # noqa: E731
    t0 = time.time()
    vh, rh, dm, pr = simulate(tr, pol, vpol, pf)
    D = tr.D
    return {"trace": tname, "policy": pol.label, "spec": repr(spec), "vram": None if vspec is None else repr(vspec),
            "pf": pf is not None, "vh": vh.sum() / D, "rh": rh.sum() / D, "dm": dm.sum() / D, "pr": pr.sum() / D,
            "dm2": dm[D // 2:].sum() / (D - D // 2), "rh2": rh[D // 2:].sum() / (D - D // 2),
            "vh2": vh[D // 2:].sum() / (D - D // 2), "sec": time.time() - t0}


def main():
    from multiprocessing import Pool
    jobs = [(t, s, None, None) for s in specs() for t in TRACES]
    with Pool(min(30, os.cpu_count())) as pool:
        res = pool.map(job, jobs, chunksize=1)
    with open(os.path.join(OUT, "results_ram.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    by = {}
    for r in res:
        by.setdefault(r["policy"], {})[r["trace"]] = r
    lines = ["%-52s | %-30s | %-30s" % ("RAM policy (VRAM = LRU)", "code: disk  RAM  VRAM  (2nd-half disk)",
                                         "uk: disk  RAM  VRAM  (2nd-half disk)")]
    for pol, d in sorted(by.items(), key=lambda kv: kv[1]["code"]["dm"] + kv[1]["uk"]["dm"]):
        lines.append("%-52s | %6.2f %6.1f %6.1f (%5.2f)     | %6.2f %6.1f %6.1f (%5.2f)" % (
            pol, d["code"]["dm"], d["code"]["rh"], d["code"]["vh"], d["code"]["dm2"],
            d["uk"]["dm"], d["uk"]["rh"], d["uk"]["vh"], d["uk"]["dm2"]))
    txt = "\n".join(lines)
    print(txt)
    with open(os.path.join(OUT, "results_ram.txt"), "w") as f:
        f.write(txt + "\n")


if __name__ == "__main__":
    main()

"""tools/glm_profile.py - GLM-5.3-Flash's expert profile from real routing (strata-glm --routes).

    python tools/glm_profile.py split --corpus DIR --tokenizer tokenizer.json --out WORK [--seg 4096]
    python tools/glm_profile.py run   --out WORK --exe strata-glm.exe --pack PACK [--chunk 2048] [--budget 60]
    python tools/glm_profile.py stats --out WORK [--vram 1310] [--ram 8300] [--profile expert-profile.bin]

split: every corpus file (*.txt, one domain each) into token segments WORK/<domain>_<i>.txt.
run:   the segments without routes yet, one engine run each, until --budget seconds have passed (rerun to go on;
       a file WORK/STOP stops it between segments). Each run's routes land in <segment>.routes.
stats: the routed mass the top-N (layer, expert) pairs cover (N = the VRAM and VRAM + RAM tiers), the same with
       the ranking taken from the other domains (how well a profile carries over), and how many distinct experts
       a window of W consecutive tokens routes to (what a speculative window reads). --profile writes the ranking
       in Strata's STRP format (tools/make_profile.py) over the 42 MoE layers numbered from 0, as experts.bin is.
"""
import argparse
import pathlib
import struct
import subprocess
import sys
import time

import numpy as np

N_LAYER, N_EXPERT, K, LEAD = 45, 288, 8, 3


def split(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in sorted(pathlib.Path(a.corpus).glob("*.txt")):
        ids = tok.encode(f.read_text(encoding="utf-8"), add_special_tokens=False).ids
        n = 0
        for i in range(0, len(ids), a.seg):
            seg = ids[i:i + a.seg]
            if len(seg) < a.seg // 4:
                break
            (out / ("%s_%02d.txt" % (f.stem, n))).write_text(" ".join(map(str, seg)))
            n += 1
        print("%-12s %6d tokens -> %d segments" % (f.stem, len(ids), n))


def run(a):
    out = pathlib.Path(a.out)
    todo = [p for p in sorted(out.glob("*_[0-9][0-9].txt")) if not p.with_suffix(".routes").exists()]
    t0 = time.time()
    for p in todo:
        if (out / "STOP").exists():
            print("STOP file found")
            break
        if time.time() - t0 > a.budget:
            break
        n = len(p.read_text().split())
        part = p.with_suffix(".routes.part")
        part.unlink(missing_ok=True)
        t = time.time()
        r = subprocess.run([str(pathlib.Path(a.exe).resolve()), "--pack", a.pack, "--tokens", str(p), "--max-new", "0", "--chunk", str(a.chunk),
                            "--max-context", str(n + 16), "--routes", str(part)], capture_output=True, text=True)
        if r.returncode != 0:
            sys.stderr.write(r.stderr[-2000:])
            raise SystemExit("%s: strata-glm exited %d" % (p.name, r.returncode))
        part.rename(p.with_suffix(".routes"))
        print("%s: %d tokens in %.1f s" % (p.name, n, time.time() - t), flush=True)
    left = sum(1 for p in out.glob("*_[0-9][0-9].txt") if not p.with_suffix(".routes").exists())
    print("%d segments left" % left)


def read_routes(path):
    """{layer: [T, K] ids} with the chunks of each layer concatenated in order."""
    blob = np.fromfile(path, dtype=np.int32)
    per, off = {}, 0
    while off + 2 <= len(blob):
        layer, n = int(blob[off]), int(blob[off + 1])
        per.setdefault(layer, []).append(blob[off + 2:off + 2 + n].reshape(-1, K))
        off += 2 + n
    return {l: np.concatenate(v) for l, v in per.items()}


def coverage(rank_counts, eval_counts, n):
    """the share of eval_counts' routed mass held by rank_counts' top-n pairs"""
    order = np.argsort(-rank_counts.ravel(), kind="stable")[:n]
    return eval_counts.ravel()[order].sum() / max(eval_counts.sum(), 1)


def stats(a):
    out = pathlib.Path(a.out)
    doms, windows = {}, {w: [0, 0] for w in (1, 2, 3, 4, 6, 8)}
    reuse = [0, 0]
    for p in sorted(out.glob("*.routes")):
        dom = p.stem.rsplit("_", 1)[0]
        c = doms.setdefault(dom, np.zeros((N_LAYER, N_EXPERT), np.int64))
        for l, ids in read_routes(p).items():
            np.add.at(c[l], ids.ravel(), 1)
            T = len(ids)
            onehot = np.zeros((T, N_EXPERT), bool)
            onehot[np.arange(T)[:, None], ids] = True
            reuse[0] += (onehot[1:] & onehot[:-1]).sum()
            reuse[1] += (T - 1) * K
            cs = np.concatenate([np.zeros((1, N_EXPERT), np.int32), np.cumsum(onehot, 0, dtype=np.int32)])
            for w in windows:
                if T >= w:
                    distinct = ((cs[w:] - cs[:-w]) > 0).sum(1)
                    windows[w][0] += distinct.sum()
                    windows[w][1] += len(distinct)
    if not doms:
        raise SystemExit("no routes in %s" % out)
    tot = sum(doms.values())
    used = (tot > 0).sum()
    print("tokens routed: %s; pairs used %d of %d" % (
        ", ".join("%s %d" % (d, c[LEAD].sum() // K) for d, c in doms.items()), used, (N_LAYER - LEAD) * N_EXPERT))
    tiers = [a.vram, a.vram + a.ram]
    print("\nrouted mass held by the top-N pairs (N = VRAM tier, VRAM + RAM tiers):")
    print("  %-26s %8s %8s   SSD" % ("ranking -> evaluated on", *("N=%d" % n for n in tiers)))
    rows = [("all -> all", tot, tot)]
    if len(doms) > 1:
        rows += [("others -> %s" % d, tot - c, c) for d, c in doms.items()]
        rows += [("%s -> %s" % (d, d), c, c) for d, c in doms.items()]
    for name, rk, ev in rows:
        cov = [coverage(rk, ev, n) for n in tiers]
        print("  %-26s %7.1f%% %7.1f%%  %5.1f%%" % (name, 100 * cov[0], 100 * cov[1], 100 * (1 - cov[1])))
    per_layer = np.sort(tot[LEAD:], axis=1)[:, ::-1]
    share = per_layer.cumsum(1) / per_layer.sum(1, keepdims=True)
    print("\nper layer, share of routed mass in its top 31 / 72 / 144 experts (VRAM-sized, quarter, half):")
    for l in range(0, N_LAYER - LEAD, 6):
        print("  layer %2d: %5.1f%% %5.1f%% %5.1f%%" % (l + LEAD, 100 * share[l, 30], 100 * share[l, 71], 100 * share[l, 143]))
    print("\ndistinct experts per layer a window of W consecutive tokens routes to (of 8W):")
    for w, (s, n) in windows.items():
        print("  W=%d: %.2f (%.0f%% of 8W)" % (w, s / n, 100 * s / n / (K * w)))
    print("expert reuse from the previous token: %.1f%%" % (100 * reuse[0] / reuse[1]))
    if a.profile:   # MoE layers numbered from 0, as in experts.bin; ties (unrouted pairs) interleaved over layers
        moe = tot[LEAD:]
        nl = moe.shape[0]
        in_layer = np.argsort(np.argsort(-moe, axis=1, kind="stable"), axis=1)
        layer = np.repeat(np.arange(nl)[:, None], N_EXPERT, 1)
        order = np.lexsort((layer.ravel(), in_layer.ravel(), -moe.ravel()))
        ranked = [(int(i // N_EXPERT), int(i % N_EXPERT)) for i in order]
        table = np.full((nl, N_EXPERT), -1, np.int32)
        for slot, (l, e) in enumerate(ranked):
            table[l, e] = slot
        with open(a.profile, "wb") as f:
            f.write(b"STRP" + struct.pack("<5I", 1, nl, N_EXPERT, len(ranked), len(ranked)))
            for l, e in ranked:
                f.write(struct.pack("<HH", l, e))
            f.write(table.tobytes())
        print("\nwrote %s (%d ranked pairs)" % (a.profile, len(ranked)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("split")
    s.add_argument("--corpus", required=True)
    s.add_argument("--tokenizer", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--seg", type=int, default=4096)
    r = sub.add_parser("run")
    r.add_argument("--out", required=True)
    r.add_argument("--exe", required=True)
    r.add_argument("--pack", required=True)
    r.add_argument("--chunk", type=int, default=2048)
    r.add_argument("--budget", type=float, default=60)
    t = sub.add_parser("stats")
    t.add_argument("--out", required=True)
    t.add_argument("--vram", type=int, default=1310)
    t.add_argument("--ram", type=int, default=8300)
    t.add_argument("--profile")
    a = ap.parse_args()
    {"split": split, "run": run, "stats": stats}[a.cmd](a)


if __name__ == "__main__":
    main()

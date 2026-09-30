"""tools/glm_compare.py - strata-glm against tools/glm_ref.py: per-layer stream error and the logits' KL.

    python tools/glm_compare.py <ref dump dir> <ref logits.f32> <engine dump dir> <engine logits.f32>
"""
import math
import pathlib
import sys

import numpy as np

rd, rl, ed, el = (pathlib.Path(a) for a in sys.argv[1:5])
worst = 0.0
for f in sorted(rd.glob("l*.f32")):
    a = np.fromfile(f, dtype=np.float32)
    g = ed / f.name
    if not g.exists():
        print("%s: no engine dump" % f.name)
        continue
    b = np.fromfile(g, dtype=np.float32)
    rel = float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-30))
    worst = max(worst, rel)
    print("%s  rel %.2e  max|d| %.3e  |ref| %.3e" % (f.stem, rel, float(np.abs(a - b).max()), float(np.abs(a).max())))
a = np.fromfile(rl, dtype=np.float32).astype(np.float64)
b = np.fromfile(el, dtype=np.float32).astype(np.float64)


def ls(x):
    m = x.max()
    return x - m - math.log(np.exp(x - m).sum())


la, lb = ls(a), ls(b)
kl = float((np.exp(la) * (la - lb)).sum())
print("logits: KL(ref||engine) %.6f  top-1 ref %d engine %d %s  top-5 ref %s engine %s"
      % (kl, int(a.argmax()), int(b.argmax()), "same" if a.argmax() == b.argmax() else "DIFF",
         list(np.argsort(-a)[:5]), list(np.argsort(-b)[:5])))
print("worst layer rel %.2e" % worst)

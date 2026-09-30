"""tools/glm_ref.py - an FP32 reference forward of GLM-5.3-Flash (NVFP4 checkpoint), one layer at a time.

    python tools/glm_ref.py --model <GLM-5.3-Flash-NVFP4 dir> --tokens <ids file> --out <logits.f32> [--dump-dir d]

The math follows transformers' modeling_glm5_next.py (main branch), written out for one sequence without padding:
mHC (Sinkhorn), KDA (recurrent form, exact), MLA NoPE with the k-pool DSA indexer, sigmoid+bias routing, the
clamped SwiGLU experts, the 3 leading dense layers. Weights are read from the safetensors files per layer and
kept in FP32 on the GPU only while that layer runs; the routed NVFP4 experts are decoded exactly (E2M1 x E4M3 x
weight_scale_2), only those the tokens route to. Memory stays flat: a layer's weights plus the activations.

Writes the last position's logits (float32, n_vocab) to --out; --dump-dir also writes each layer's output streams.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from glm_pack import Checkpoint, e4m3  # noqa: E402

DEV = "cuda"
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)


class W:
    """Tensors of the checkpoint as FP32 on the GPU."""

    def __init__(self, ck: Checkpoint):
        self.ck = ck
        self.e2m1 = E2M1.to(DEV)
        self.e4m3_lut = torch.from_numpy(e4m3(np.arange(256, dtype=np.uint8))).to(DEV)

    def t(self, name: str) -> torch.Tensor:
        f, off, n, dt, shape = self.ck.info(name)
        raw = np.asarray(self.ck.raw(name))
        if dt == "BF16":
            x = torch.from_numpy(raw.view(np.int16).copy()).view(torch.bfloat16)
        elif dt == "F32":
            x = torch.from_numpy(raw.view(np.float32).copy())
        elif dt == "U8" and name.endswith(".weight"):   # NVFP4 (the leading dense MLP layers 0-2): decoded exactly
            return self.nvfp4(name[:-len(".weight")], shape[0], shape[1] * 2)
        else:
            raise ValueError("%s: dtype %s" % (name, dt))
        return x.reshape(shape).to(DEV).float()

    def nvfp4(self, prefix: str, rows: int, cols: int) -> torch.Tensor:
        w = torch.from_numpy(np.asarray(self.ck.raw(prefix + ".weight")).copy()).to(DEV)
        s = torch.from_numpy(np.asarray(self.ck.raw(prefix + ".weight_scale")).copy()).to(DEV)
        s2 = float(np.frombuffer(np.asarray(self.ck.raw(prefix + ".weight_scale_2")).tobytes(), dtype=np.float32)[0])
        codes = torch.stack([w & 0x0F, w >> 4], dim=-1).reshape(rows, cols).long()
        scale = self.e4m3_lut[s.long()].reshape(rows, cols // 16).repeat_interleave(16, dim=1)
        return self.e2m1[codes] * scale * s2


def rms(x, w=None, eps=1e-5):
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    return y * w if w is not None else y


def hc_pre(streams, fn, base, scale, cfg):
    T, N, D = streams.shape
    flat = rms(streams.reshape(T, N * D), None, cfg["rms_norm_eps"])
    mix = flat @ fn.t()                                         # [T, 24]
    pre_w, post_w, comb_w = mix.split([N, N, N * N], dim=-1)
    pre_b, post_b, comb_b = base.split([N, N, N * N])
    eps = cfg["hc_eps"]
    pre = torch.sigmoid(pre_w * scale[0] + pre_b) + eps
    post = 2 * torch.sigmoid(post_w * scale[1] + post_b)
    comb = torch.softmax((comb_w * scale[2] + comb_b).view(T, N, N), dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(cfg["hc_sinkhorn_iters"] - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    collapsed = (pre.unsqueeze(-1) * streams).sum(dim=1)
    return post, comb, collapsed


def hc_post(y, residual, post, comb):
    # out[n] = post[n] * y + sum_m comb[m][n] * residual[m]
    return post.unsqueeze(-1) * y.unsqueeze(1) + torch.einsum("tmn,tmd->tnd", comb, residual)


def swiglu(x, wg, wu, wd, limit):
    g = (x @ wg.t()).clamp(max=limit)
    u = (x @ wu.t()).clamp(min=-limit, max=limit)
    return (F.silu(g) * u) @ wd.t()


def kda(x, w: W, p, cfg):
    T = x.shape[0]
    la = cfg["linear_attn_config"]
    H, Dh, K = la["num_heads"], la["head_dim"], la["short_conv_kernel_size"]
    outs = []
    for proj, conv in (("q_proj", "q_conv1d"), ("k_proj", "k_conv1d"), ("v_proj", "v_conv1d")):
        y = x @ w.t(p + proj + ".weight").t()                   # [T, H*Dh]
        cw = w.t(p + conv + ".weight").view(H * Dh, K)          # causal depthwise conv, zero history
        yp = F.pad(y.t(), (K - 1, 0))                            # [C, T+K-1]
        c = sum(cw[:, j:j + 1] * yp[:, j:j + T] for j in range(K))
        outs.append(F.silu(c.t()).view(T, H, Dh))
    q, k, v = outs
    q = q / torch.sqrt((q * q).sum(-1, keepdim=True) + 1e-6) * (Dh ** -0.5)
    k = k / torch.sqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    g = (x @ w.t(p + "f_a_proj.weight").t()) @ w.t(p + "f_b_proj.weight").t() + w.t(p + "dt_bias")
    decay = torch.exp(w.t(p + "A_log")).view(1, H, 1)
    g = la["gate_lower_bound"] * torch.sigmoid(decay * g.view(T, H, Dh))       # log-decay per key channel
    beta = torch.sigmoid(x @ w.t(p + "b_proj.weight").t())                      # [T, H]
    S = torch.zeros(H, Dh, Dh, device=DEV)
    o = torch.empty(T, H, Dh, device=DEV)
    for t in range(T):
        S = S * torch.exp(g[t]).unsqueeze(-1)
        kv_mem = (S * k[t].unsqueeze(-1)).sum(dim=-2)
        delta = (v[t] - kv_mem) * beta[t].unsqueeze(-1)
        S = S + k[t].unsqueeze(-1) * delta.unsqueeze(-2)
        o[t] = (S * q[t].unsqueeze(-1)).sum(dim=-2)
    gate = ((x @ w.t(p + "g_a_proj.weight").t()) @ w.t(p + "g_b_proj.weight").t()).view(T, H, Dh)
    o = rms(o, w.t(p + "o_norm.weight"), cfg["rms_norm_eps"]) * torch.sigmoid(gate)
    return o.reshape(T, H * Dh) @ w.t(p + "o_proj.weight").t()


def dsa(x, w: W, p, cfg):
    T = x.shape[0]
    Hn, dk, dv, R = cfg["num_attention_heads"], cfg["qk_nope_head_dim"], cfg["v_head_dim"], cfg["kv_lora_rank"]
    eps = cfg["rms_norm_eps"]
    q_resid = rms(x @ w.t(p + "q_a_proj.weight").t(), w.t(p + "q_a_layernorm.weight"), eps)
    q = (q_resid @ w.t(p + "q_b_proj.weight").t()).view(T, Hn, dk)
    lat = rms(x @ w.t(p + "kv_a_proj_with_mqa.weight").t(), w.t(p + "kv_a_layernorm.weight"), eps)   # [T, 512]
    kv = (lat @ w.t(p + "kv_b_proj.weight").t()).view(T, Hn, dk + dv)
    kk, vv = kv.split([dk, dv], dim=-1)

    # the indexer: pools of kpool tokens from position 0 (one sequence, no padding)
    ip = p + "indexer."
    ih, idim, kpool, topk = cfg["index_n_heads"], cfg["index_head_dim"], cfg["index_kpool"], cfg["index_topk"]
    iq = (q_resid @ w.t(ip + "wq_b.weight").t()).view(T, ih, idim)
    ik = F.layer_norm(x @ w.t(ip + "wk.weight").t(), (idim,), w.t(ip + "k_norm.weight"), w.t(ip + "k_norm.bias"), 1e-6)
    gs = x @ w.t(ip + "index_kpool_compress_gate").t()
    n_pool = T // kpool
    sel = torch.zeros(T, T, dtype=torch.bool, device=DEV)
    if n_pool > 0:
        gk = ik[: n_pool * kpool].view(n_pool, kpool, idim)
        gl = gs[: n_pool * kpool].view(n_pool, kpool, idim) + w.t(ip + "index_kpool_compress_ape").unsqueeze(0)
        pooled = (torch.softmax(gl, dim=1) * gk).sum(dim=1)                                   # [P, idim]
        scores = F.relu(torch.einsum("thd,pd->thp", iq, pooled) * idim ** -0.5)
        wts = (x @ w.t(ip + "weights_proj.weight").t()) * ih ** -0.5                          # [T, ih]
        isc = torch.einsum("th,thp->tp", wts, scores)
        pool_end = torch.arange(n_pool, device=DEV) * kpool + kpool - 1
        visible = pool_end.unsqueeze(0) <= torch.arange(T, device=DEV).unsqueeze(1)          # [T, P]
        isc = isc.masked_fill(~visible, torch.finfo(isc.dtype).min)
        kk_sel = min(topk // kpool, n_pool)
        chosen = isc.topk(kk_sel, dim=-1).indices                                             # [T, kk]
        ok = visible.gather(1, chosen)
        for j in range(kpool):
            idx = chosen * kpool + j
            sel.scatter_(1, idx, ok | sel.gather(1, idx))
    # the incomplete tail of each query's visible tokens
    tpos = torch.arange(T, device=DEV)
    tail_count = (tpos + 1) % kpool
    for j in range(kpool - 1):
        idx = tpos + 1 - tail_count + j
        live = (j < tail_count) & (idx < T)
        sel[tpos[live], idx[live]] = True
    sel &= torch.ones(T, T, dtype=torch.bool, device=DEV).tril()

    att = torch.einsum("thd,shd->hts", q, kk) * (dk ** -0.5)
    att = att.masked_fill(~sel.unsqueeze(0), float("-inf")).softmax(dim=-1)
    o = torch.einsum("hts,shd->thd", att, vv).reshape(T, Hn * dv)
    return o @ w.t(p + "o_proj.weight").t()


def moe(x, w: W, p, cfg, layer):
    T = x.shape[0]
    lim = cfg["swiglu_limit"]
    logits = x @ w.t(p + "gate.weight").t()
    scores = torch.sigmoid(logits)
    choice = scores + w.t(p + "gate.e_score_correction_bias")
    top = choice.topk(cfg["num_experts_per_tok"], dim=-1).indices                    # [T, 8]
    wt = scores.gather(1, top)
    wt = wt / (wt.sum(-1, keepdim=True) + 1e-20) * cfg["routed_scaling_factor"]
    out = torch.zeros_like(x)
    E, Hd, Ff = cfg["n_routed_experts"], cfg["hidden_size"], cfg["moe_intermediate_size"]
    for e in top.unique().tolist():
        rows, slot = (top == e).nonzero(as_tuple=True)
        pe = p + "experts.%d." % e
        y = swiglu(x[rows], w.nvfp4(pe + "gate_proj", Ff, Hd), w.nvfp4(pe + "up_proj", Ff, Hd),
                   w.nvfp4(pe + "down_proj", Hd, Ff), lim)
        out.index_add_(0, rows, y * wt[rows, slot].unsqueeze(-1))
    sp = p + "shared_experts."
    out += swiglu(x, w.t(sp + "gate_proj.weight"), w.t(sp + "up_proj.weight"), w.t(sp + "down_proj.weight"), lim)
    return out, top


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", required=True, help="token ids, whitespace or comma separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump-dir")
    ap.add_argument("--layers", type=int, default=0, help="stop after this many layers (debugging)")
    ap.add_argument("--topk", type=int, default=0, help="override index_topk (a huge value: dense attention)")
    a = ap.parse_args()
    d = pathlib.Path(a.model)
    cfg = json.loads((d / "config.json").read_text())
    cfg = cfg.get("text_config", cfg)
    if a.topk:
        cfg["index_topk"] = a.topk
    ids = [int(t) for t in pathlib.Path(a.tokens).read_text().replace(",", " ").split()]
    ck = Checkpoint(d)
    w = W(ck)
    P = "model.language_model."
    torch.backends.cuda.matmul.allow_tf32 = False
    T, N = len(ids), cfg["hc_mult"]
    t0 = time.time()
    emb = w.t(P + "embed_tokens.weight")[torch.tensor(ids, device=DEV)]
    torch.cuda.empty_cache()
    streams = emb.unsqueeze(1).expand(T, N, -1).contiguous()
    n_layers = a.layers or cfg["num_hidden_layers"]
    routes = {}
    for l in range(n_layers):
        p = P + "layers.%d." % l
        post, comb, x = hc_pre(streams, w.t(p + "hc_attn_fn"), w.t(p + "hc_attn_base"), w.t(p + "hc_attn_scale"), cfg)
        x = rms(x, w.t(p + "input_layernorm.weight"), cfg["rms_norm_eps"])
        if cfg["layer_types"][l] == "linear_attention":
            y = kda(x, w, p + "self_attn.", cfg)
        else:
            y = dsa(x, w, p + "self_attn.", cfg)
        streams = hc_post(y, streams, post, comb)
        post, comb, x = hc_pre(streams, w.t(p + "hc_ffn_fn"), w.t(p + "hc_ffn_base"), w.t(p + "hc_ffn_scale"), cfg)
        x = rms(x, w.t(p + "post_attention_layernorm.weight"), cfg["rms_norm_eps"])
        if cfg["mlp_layer_types"][l] == "dense":
            mp = p + "mlp."
            y = swiglu(x, w.t(mp + "gate_proj.weight"), w.t(mp + "up_proj.weight"), w.t(mp + "down_proj.weight"),
                       cfg["swiglu_limit"])
        else:
            y, top = moe(x, w, p + "mlp.", cfg, l)
            routes[l] = top.cpu().numpy()
        streams = hc_post(y, streams, post, comb)
        if a.dump_dir:
            pathlib.Path(a.dump_dir).mkdir(parents=True, exist_ok=True)
            streams[-1].cpu().numpy().astype(np.float32).tofile(pathlib.Path(a.dump_dir) / ("l%02d.f32" % l))
        ck.maps.clear()
        torch.cuda.empty_cache()
        print("layer %2d %-18s %.1f s" % (l, cfg["layer_types"][l], time.time() - t0), flush=True)
    h = rms(streams.mean(dim=1), w.t(P + "norm.weight"), cfg["rms_norm_eps"])
    logits = h[-1] @ w.t("lm_head.weight").t()
    logits.cpu().numpy().astype(np.float32).tofile(a.out)
    top5 = torch.topk(logits, 5)
    print("done in %.0f s; last position's top-5: %s" % (time.time() - t0, list(zip(top5.indices.tolist(),
          [round(v, 3) for v in top5.values.tolist()]))), flush=True)
    if a.dump_dir:
        np.save(pathlib.Path(a.dump_dir) / "routes.npy", np.stack([routes[l] for l in sorted(routes)]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

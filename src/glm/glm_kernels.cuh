// src/glm/glm_kernels.cuh - GLM-5.3-Flash's own kernels for strata-glm (M1: FP32 math, BF16 weights, one token).
//
// Every formula follows transformers' modeling_glm5_next.py; tools/glm_ref.py is the FP32 reference they are
// checked against layer by layer.
#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace glm {

using bf16 = uint16_t;   // raw BF16 bits (weights as the checkpoint stores them)

/// y[t][r] = sum_c W[r][c] * x[t][c] for t < nt (nt <= 8); W is BF16 [rows][cols] (cols % 8 == 0), x and y FP32.
/// `x_ld` / `y_ld`: row strides of x and y (0 = cols / rows).
void gemv_bf16(const bf16* W, const float* x, float* y, int rows, int cols, int nt, cudaStream_t s,
               int x_ld = 0, int y_ld = 0);
/// The same with FP32 weights (the leading dense MLP layers, decoded from NVFP4 at load).
void gemv_f32(const float* W, const float* x, float* y, int rows, int cols, int nt, cudaStream_t s);

/// out = x * rsqrt(mean(x^2) + eps) * w (w BF16, or none) over rows of n; nrows rows.
void rmsnorm(const float* x, const bf16* w, float eps, float* out, int n, int nrows, cudaStream_t s);
/// LayerNorm with BF16 weight and bias (the indexer's k_norm).
void layernorm(const float* x, const bf16* w, const bf16* b, float eps, float* out, int n, cudaStream_t s);

/// mHC pre: from the 4 streams (hc x n) and the raw mixes `mix` (24 = fn . streams, before the input norm's
/// 1/rms), writes collapsed x (n), post (4) and comb (4x4, [m][n] = input stream m into output stream n).
void hc_pre_finish(const float* streams, const float* mix, const bf16* base, const bf16* scale, int n, float rms_eps,
                   float hc_eps, int sinkhorn_iters, float* x, float* post, float* comb, cudaStream_t s);
/// streams_out[j] = post[j] * y + sum_m comb[m][j] * streams_in[m] (may alias streams_in: it is read first).
void hc_post(const float* y, const float* streams_in, const float* post, const float* comb, float* streams_out, int n,
             cudaStream_t s);
/// out = mean over the 4 streams
void hc_mean(const float* streams, float* out, int n, cudaStream_t s);

/// Causal depthwise conv (kernel 4) + SiLU for one token: window = [state[c][0..2], x[c]], state shifts.
/// w: FP32 [C][4]; C channels.
void conv_silu_step(const float* x, const float* w, float* state, float* out, int C, cudaStream_t s);
/// KDA gates: q,k l2-normalized per head (q also times dh^-0.5); g = lb * sigmoid(exp(A_log[h]) * (gf + dt_bias));
/// beta = sigmoid(b). q, k: [H][dh] in place; gf: [H*dh] in, g out (log-decay); b: [H] in, beta out.
void kda_prep(float* q, float* k, float* gf, const float* dt_bias, const float* A_log, float lower_bound, float* b,
              int H, int dh, cudaStream_t s);
/// One token of the gated delta rule per head: S[h] (dh x dh, [key][value]) *= exp(g) per key row; kv = S^T k;
/// delta = beta * (v - kv); S += k (x) delta; o = S^T q.
void kda_step(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta, float* o,
              int H, int dh, cudaStream_t s);
/// o[h] = rmsnorm(o[h]) * w * sigmoid(gate[h]) per head (the gated output norm).
void kda_out_norm(float* o, const bf16* w, const float* gate, float eps, int H, int dh, cudaStream_t s);

/// MLA, absorbed: qa[h][c] = sum_{d<dk} q[h][d] * kvb[h*(dk+dv) + d][c]   (kvb BF16 [H*(dk+dv)][R])
void mla_absorb_q(const float* q, const bf16* kvb, float* qa, int H, int dk, int dv, int R, cudaStream_t s);
/// scores over the selected cache rows sel[0..n_sel) (latent cache [pos][R]), softmax, ctx[h] = sum p * lat.
void mla_attend(const float* qa, const float* lat, const int32_t* sel, int n_sel, float scale, float* ctx, int H,
                int R, cudaStream_t s);
/// out[h][e] = sum_c kvb[h*(dk+dv) + dk + e][c] * ctx[h][c]
void mla_value(const float* ctx, const bf16* kvb, float* out, int H, int dk, int dv, int R, cudaStream_t s);

/// pooled[d] = sum_j softmax_j(gate[j][d] + ape[j][d]) * key[j][d] over the pool's kpool rows
void idx_pool(const float* key, const float* gate, const bf16* ape, float* pooled, int kpool, int dim, cudaStream_t s);
/// score[p] = H^-0.5 * sum_h w[h] * relu(q[h] . pooled[p] * dim^-0.5)  for p < n_pool   (w = weights_proj(x))
void idx_scores(const float* q, const float* w, const float* pooled, float* score, int n_pool, int H, int dim,
                cudaStream_t s);

/// Router: scores = sigmoid(logits); pick top-k of scores + bias; weights = scores / sum * scaling. One token.
void route_topk(const float* logits, const float* bias, int n_expert, int k, float scaling, int32_t* ids, float* wts,
                cudaStream_t s);
/// h = silu(min(g, lim)) * clamp(u, -lim, lim); g and u interleaved as two arrays of n
void swiglu_clamp(const float* g, const float* u, float* h, int n, float lim, cudaStream_t s);
/// y += sum_e w[e] * rows[e] over k rows of n (the routed experts' outputs)
void combine_rows(const float* rows, const float* w, int k, float* y, int n, cudaStream_t s);
/// out[i] = a[i] + b[i]
void add(const float* a, const float* b, float* out, int n, cudaStream_t s);

/// NVFP4 (ModelOpt: u8 [rows][cols/2], e4m3 [rows][cols/16], scale2) -> FP32 [rows][cols], exactly
void nvfp4_to_f32(const uint8_t* w, const uint8_t* sc, float s2, float* out, int rows, int cols, cudaStream_t s);
/// BF16 row of the embedding table (host or device) -> FP32
void bf16_to_f32(const bf16* in, float* out, int n, cudaStream_t s);

}  // namespace glm

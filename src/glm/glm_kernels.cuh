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
using f16 = uint16_t;    // raw FP16 bits (the MLA latent cache)

/// y[t][r] = sum_c W[r][c] * x[t][c] for t < nt (nt <= 8); W is BF16 [rows][cols] (cols % 8 == 0), x and y FP32.
/// `x_ld` / `y_ld`: row strides of x and y (0 = cols / rows).
void gemv_bf16(const bf16* W, const float* x, float* y, int rows, int cols, int nt, cudaStream_t s,
               int x_ld = 0, int y_ld = 0);
/// The same with FP32 weights (the leading dense MLP layers, decoded from NVFP4 at load).
void gemv_f32(const float* W, const float* x, float* y, int rows, int cols, int nt, cudaStream_t s);

/// A dense weight matrix [rows][cols]: BF16, or FP8 E4M3 with an FP32 scale per row. A BF16 matrix may carry
/// the scale too (the leading dense MLP: its NVFP4 values held exactly in BF16, weight_scale_2 on the output).
struct Mat {
    const void* w = nullptr;
    const float* scale = nullptr;   ///< per row (output), or null
    bool fp8 = false;
    const bf16* b16() const { return (const bf16*) w; }
};
/// y[t][r] = scale[r] * sum_c W[r][c] * x[t][c], as gemv_bf16 (cols % 16 == 0 for FP8).
void gemv(const Mat& W, const float* x, float* y, int rows, int cols, int nt, cudaStream_t s, int x_ld = 0, int y_ld = 0);
/// W (BF16) -> q (E4M3) and scale[r] = amax(row r) / 448, once at load.
void quant_fp8_rows(const bf16* w, int rows, int cols, uint8_t* q, float* scale, cudaStream_t s);
/// E4M3 -> BF16, exactly (every E4M3 value is a BF16 value); the prompt path's GEMMs read the result.
void fp8_to_bf16(const uint8_t* q, bf16* out, size_t n, cudaStream_t s);
/// Y[t][n] *= scale[n] for t < T (row stride ldy).
void scale_cols(float* Y, const float* scale, int N, int T, int ldy, cudaStream_t s);
/// NVFP4 (codes, E4M3 block scales) -> BF16 without weight_scale_2: e2m1 x e4m3 has at most 5 significant bits.
void nvfp4_to_bf16(const uint8_t* w, const uint8_t* sc, bf16* out, int rows, int cols, cudaStream_t s);
void fill(float* p, float v, int n, cudaStream_t s);

/// out = x * rsqrt(mean(x^2) + eps) * w (w BF16, or none) over rows of n; nrows rows.
void rmsnorm(const float* x, const bf16* w, float eps, float* out, int n, int nrows, cudaStream_t s);
/// the same into FP16 (the MLA latent cache)
void rmsnorm_f16(const float* x, const bf16* w, float eps, f16* out, int n, int nrows, cudaStream_t s);
/// the same into int8 with a float scale per 64 values (n % 64 == 0): out [nrows][n], scale [nrows][n / 64]
void rmsnorm_i8(const float* x, const bf16* w, float eps, int8_t* out, float* scale, int n, int nrows, cudaStream_t s);
/// LayerNorm with BF16 weight and bias (the indexer's k_norm).
void layernorm(const float* x, const bf16* w, const bf16* b, float eps, float* out, int n, cudaStream_t s);

/// mHC pre: from the 4 streams (hc x n) and the raw mixes `mix` (24 = fn . streams, before the input norm's
/// 1/rms), writes collapsed x (n), post (4) and comb (4x4, [m][n] = input stream m into output stream n).
void hc_pre_finish(const float* streams, const float* mix, const bf16* base, const bf16* scale, int n, float rms_eps,
                   float hc_eps, int sinkhorn_iters, float* x, float* post, float* comb, cudaStream_t s,
                   const bf16* norm_w = nullptr, float* xn = nullptr, int mix_parts = 1);
/// the 24 mixes as `parts` partial sums over column slices ([parts][24], summed by hc_pre_finish(mix_parts)):
/// 16 blocks instead of the 3 a 24-row gemv gets
void hc_mix(const bf16* W, const float* x, float* parts, int rows, int cols, int nparts, cudaStream_t s);
/// (with norm_w and xn: also xn = rmsnorm(x) * norm_w, eps rms_eps - the layer's input norm in the same launch)
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


// ---------------------------------------------------------------- the prompt path: T tokens at once
/// LayerNorm over nrows rows of n
void layernorm_rows(const float* x, const bf16* w, const bf16* b, float eps, float* out, int n, int nrows, cudaStream_t s);
/// hc_pre_finish for T tokens: streams [T][4n], mix [T][24] -> x [T][n], post [T][4], comb [T][16]
void hc_pre_rows(const float* streams, const float* mix, const bf16* base, const bf16* scale, int n, float rms_eps,
                 float hc_eps, int iters, float* x, float* post, float* comb, int T, cudaStream_t s);
/// hc_post for T tokens (in place allowed)
void hc_post_rows(const float* y, const float* streams_in, const float* post, const float* comb, float* streams_out,
                  int n, int T, cudaStream_t s);
/// the causal conv + SiLU over T tokens in order (x, out: [T][C]; state [C][3] carried in and out)
void conv_silu_seq(const float* x, const float* w, float* state, float* out, int C, int T, cudaStream_t s);
/// kda_prep for T tokens (q, k, gf: [T][H*dh]; b: [T][H])
void kda_prep_rows(float* q, float* k, float* gf, const float* dt_bias, const float* A_log, float lower_bound, float* b,
                   int H, int dh, int T, cudaStream_t s);
/// the gated delta rule over T tokens in order, S [H][dh][dh] carried in and out; o [T][H*dh]
void kda_scan(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta, float* o,
              int H, int dh, int T, cudaStream_t s);
/// kda_out_norm for T tokens
void kda_out_norm_rows(float* o, const bf16* w, const float* gate, float eps, int H, int dh, int T, cudaStream_t s);
/// MLA attention for T queries at positions pos0.. : query t attends to cache rows sel[t][0..cnt[t]) (sel == null:
/// all rows 0..pos0+t, causal). qa: [T][H][R]; ctx out [T][H][R]; max_sel: the longest list (shared memory).
void mla_attend_rows(const float* qa, const float* lat, const int32_t* sel, const int32_t* cnt, int sel_ld, int pos0,
                     int max_sel, float scale, float* ctx, int H, int R, int T, cudaStream_t s);
/// the pooled keys of pools [p0, p0 + np): pooled[p] from key/gate rows 4p..4p+3
/// key / gate rows from pool p0 (relative to their window) on; the pools go to pooled rows out_p0.. (out_p0 < 0: p0)
void idx_pool_rows(const float* key, const float* gate, const bf16* ape, float* pooled, int p0, int np, int kpool, int dim,
                   cudaStream_t s, int out_p0 = -1);
/// scores [T][n_pool] for queries at positions pos0..; a pool is visible when its last token <= the query's position,
/// invisible ones get -FLT_MAX
void idx_scores_rows(const float* q, const float* w, const float* pooled, float* score, int n_pool, int H, int dim,
                     int pos0, int kpool, int T, cudaStream_t s);
/// For queries at positions pos0..pos0+T-1 with scores [T][ld] over complete pools: sel [T][sel_ld] = every
/// visible token while a query sees <= budget pools, else the budget top-scoring pools' tokens (by pool index)
/// and its incomplete tail; cnt[t] = how many. On the GPU (glm_select.cu): no host round trip.
void idx_select_rows(const float* score, int ld, int pos0, int kpool, int budget, int32_t* sel, int sel_ld, int32_t* cnt,
                     int T, cudaStream_t s);
/// The prompt path's MMQ experts (glm_moe.cu): a group of n blobs `stride` apart; rows r0..r0+nr, expert j owning
/// [r0 + rb[j], r0 + rb[j+1]) (rb relative, on the device).
/// tails: out[4j..4j+3] = blob j's {s_gate, s_up, s_down, 0}
void gather_tails(const uint8_t* base, size_t stride, size_t tail_off, int n, float* out, cudaStream_t s);
/// h[r] = silu(clamp(gate * s_gate)) * clamp(up * s_up) from MMQ's gate|up rows gu[r] (2 ff wide)
void swiglu_rows(const float* gu, float* h, const int32_t* rb, int n, const float* tails, long long r0, int nr, int ff,
                 float lim, cudaStream_t s);
/// wts[dst[r]] *= s_down of row r's expert (the combine then applies it)
void scale_entry_wts(float* wts, const int32_t* dst, const int32_t* rb, int n, const float* tails, int nr, cudaStream_t s);
/// MLA attention on tensor cores (glm_mla.cu), 64 heads x 512: ctx[t][h] over query t's keys (sel / cnt, or every
/// position <= pos0 + t when sel is null). nsplit > 1 (decode) splits the keys over blocks and needs `part`
/// (mla_tc_part_floats(T, nsplit) floats).
/// lat8 / lat8s (not null): the latent as int8 with a scale per 64 values instead of FP16 `lat`
void mla_attend_tc(const float* qa, const f16* lat, const int8_t* lat8, const float* lat8s, const int32_t* sel, const int32_t* cnt, int sel_ld, int pos0,
                   float scale, float* ctx, int T, int nsplit, float* part, cudaStream_t s);
size_t mla_tc_part_floats(int T, int nsplit);
/// conv_silu_seq's scratch (C x T floats) up front / back
void conv_silu_reserve(size_t n);
void conv_silu_release();
/// route_topk for T tokens: logits [T][n], ids / wts [T][k]
void route_rows(const float* logits, const float* bias, int n_expert, int k, float scaling, int32_t* ids, float* wts,
                int T, cudaStream_t s);
/// y[t] += sum_j w[t][j] * rows[t*k + j] over n
void combine_rows_t(const float* rows, const float* w, int k, float* y, int n, int T, cudaStream_t s);

}  // namespace glm

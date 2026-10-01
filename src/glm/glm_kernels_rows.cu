// src/glm/glm_kernels_rows.cu - the prompt path's kernels (T tokens at once); the same math as glm_kernels.cu.
#include "glm_kernels.cuh"

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace glm {
namespace {

__device__ __forceinline__ float bf(bf16 v) { return __uint_as_float((uint32_t) v << 16); }

__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ float block_sum(float v) {
    __shared__ float part[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) part[wid] = v;
    __syncthreads();
    float r = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0.f;
    if (wid == 0) r = warp_sum(r);
    if (threadIdx.x == 0) total = r;
    __syncthreads();
    return total;
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm kernel %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__global__ void layernorm_rows_kernel(const float* __restrict__ x, const bf16* __restrict__ w, const bf16* __restrict__ b,
                                      float eps, float* __restrict__ out, int n) {
    const float* xr = x + (size_t) blockIdx.x * n;
    float* o = out + (size_t) blockIdx.x * n;
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += xr[i];
    const float mean = block_sum(s) / (float) n;
    float v = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) v += (xr[i] - mean) * (xr[i] - mean);
    const float inv = rsqrtf(block_sum(v) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = (xr[i] - mean) * inv * bf(w[i]) + bf(b[i]);
}

__global__ void hc_pre_rows_kernel(const float* __restrict__ streams_all, const float* __restrict__ mix_all,
                                   const bf16* __restrict__ base, const bf16* __restrict__ scale, int n, float rms_eps,
                                   float hc_eps, int iters, float* __restrict__ x_all, float* __restrict__ post_all,
                                   float* __restrict__ comb_all) {
    const int t = blockIdx.x;
    const float* streams = streams_all + (size_t) t * 4 * n;
    const float* mix = mix_all + (size_t) t * 24;
    __shared__ float pre[4], comb[16], inv_rms;
    float ss = 0.f;
    for (int i = threadIdx.x; i < 4 * n; i += blockDim.x) ss += streams[i] * streams[i];
    ss = block_sum(ss);
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(ss / (float) (4 * n) + rms_eps);
        const float s0 = bf(scale[0]), s1 = bf(scale[1]), s2 = bf(scale[2]);
        for (int j = 0; j < 4; ++j) {
            pre[j] = 1.f / (1.f + expf(-(mix[j] * inv_rms * s0 + bf(base[j])))) + hc_eps;
            post_all[(size_t) t * 4 + j] = 2.f / (1.f + expf(-(mix[4 + j] * inv_rms * s1 + bf(base[4 + j]))));
        }
        for (int r = 0; r < 4; ++r) {
            float m = -FLT_MAX, v[4];
            for (int c = 0; c < 4; ++c) {
                v[c] = mix[8 + 4 * r + c] * inv_rms * s2 + bf(base[8 + 4 * r + c]);
                m = fmaxf(m, v[c]);
            }
            float sum = 0.f;
            for (int c = 0; c < 4; ++c) sum += (v[c] = expf(v[c] - m));
            for (int c = 0; c < 4; ++c) comb[4 * r + c] = v[c] / sum + hc_eps;
        }
        for (int it = 0; it < iters; ++it) {
            if (it > 0)
                for (int r = 0; r < 4; ++r) {
                    float sum = 0.f;
                    for (int c = 0; c < 4; ++c) sum += comb[4 * r + c];
                    for (int c = 0; c < 4; ++c) comb[4 * r + c] /= sum + hc_eps;
                }
            for (int c = 0; c < 4; ++c) {
                float sum = 0.f;
                for (int r = 0; r < 4; ++r) sum += comb[4 * r + c];
                for (int r = 0; r < 4; ++r) comb[4 * r + c] /= sum + hc_eps;
            }
        }
        for (int j = 0; j < 16; ++j) comb_all[(size_t) t * 16 + j] = comb[j];
    }
    __syncthreads();
    float* x = x_all + (size_t) t * n;
    for (int d = threadIdx.x; d < n; d += blockDim.x)
        x[d] = pre[0] * streams[d] + pre[1] * streams[n + d] + pre[2] * streams[2 * n + d] + pre[3] * streams[3 * n + d];
}

__global__ void hc_post_rows_kernel(const float* __restrict__ y_all, const float* streams_in_all,
                                    const float* __restrict__ post_all, const float* __restrict__ comb_all,
                                    float* streams_out_all, int n) {
    const int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n) return;
    const float* si = streams_in_all + (size_t) t * 4 * n;
    float* so = streams_out_all + (size_t) t * 4 * n;
    const float* post = post_all + (size_t) t * 4;
    const float* comb = comb_all + (size_t) t * 16;
    const float r0 = si[d], r1 = si[n + d], r2 = si[2 * n + d], r3 = si[3 * n + d], yd = y_all[(size_t) t * n + d];
    for (int j = 0; j < 4; ++j)
        so[j * n + d] = post[j] * yd + comb[j] * r0 + comb[4 + j] * r1 + comb[8 + j] * r2 + comb[12 + j] * r3;
}

// window [x[t-3], x[t-2], x[t-1], x[t]] with x[-3..-1] from the state: no recurrence, so every (t, c) at once
__global__ void conv_silu_par_kernel(const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ state,
                                     float* __restrict__ out, int C, int T) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) C * T) return;
    const int t = (int) (i / C), c = (int) (i % C);
    auto xin = [&](int tt) { return tt >= 0 ? x[(size_t) tt * C + c] : state[(size_t) c * 3 + 3 + tt]; };
    const float v = w[(size_t) c * 4] * xin(t - 3) + w[(size_t) c * 4 + 1] * xin(t - 2) + w[(size_t) c * 4 + 2] * xin(t - 1) +
                    w[(size_t) c * 4 + 3] * xin(t);
    out[i] = v / (1.f + expf(-v));
}

__global__ void conv_state_kernel(const float* __restrict__ x, float* __restrict__ state, int C, int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float w3[3];
    for (int j = 0; j < 3; ++j) {
        const int tt = T - 3 + j;
        w3[j] = tt >= 0 ? x[(size_t) tt * C + c] : state[(size_t) c * 3 + 3 + tt];
    }
    for (int j = 0; j < 3; ++j) state[(size_t) c * 3 + j] = w3[j];
}

__global__ void kda_prep_rows_kernel(float* __restrict__ q, float* __restrict__ k, float* __restrict__ gf,
                                     const float* __restrict__ dt_bias, const float* __restrict__ A_log, float lb,
                                     float* __restrict__ b, int H, int dh) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const size_t j = ((size_t) t * H + h) * dh + i;
    const float qn = block_sum(q[j] * q[j]), kn = block_sum(k[j] * k[j]);
    q[j] = q[j] / sqrtf(qn + 1e-6f) * rsqrtf((float) dh);
    k[j] = k[j] / sqrtf(kn + 1e-6f);
    gf[j] = lb / (1.f + expf(-(expf(A_log[h]) * (gf[j] + dt_bias[(size_t) h * dh + i]))));
    if (i == 0) b[(size_t) t * H + h] = 1.f / (1.f + expf(-b[(size_t) t * H + h]));
}

// one block per head, S (dh x dh) in shared memory, the tokens in order
// The value columns of S are independent (delta_c = beta (v_c - sum_i S[i][c] k_i) touches column c only; the
// decay scales rows), so a head spreads over 8 warps of 16 columns, two lanes a column (lane and lane + 16, 64 rows
// each, their sums joined by one shuffle) - 64 floats of state a thread stay in registers (128 went to the stack).
// The scan is a chain over tokens, so the next token's inputs are loaded while this one computes.
template <int DH>
__global__ void __launch_bounds__(32) kda_scan_kernel(float* __restrict__ S, const float* __restrict__ q,
                                                      const float* __restrict__ k, const float* __restrict__ v,
                                                      const float* __restrict__ g, const float* __restrict__ beta,
                                                      float* __restrict__ o, int H, int T) {
    constexpr int R = DH / 2;
    static_assert(DH == 128, "a 128-wide head");
    __shared__ float qs[2][DH], ks[2][DH], eg[2][DH];
    const int h = blockIdx.x, lane = threadIdx.x, c = blockIdx.y * 16 + (lane & 15), r0 = (lane >> 4) * R;
    float* Sg = S + (size_t) h * DH * DH;
    float st[R];
#pragma unroll
    for (int i = 0; i < R; ++i) st[i] = Sg[(size_t) (r0 + i) * DH + c];
    float rq0 = 0.f, rq1 = 0.f, rq2 = 0.f, rq3 = 0.f, rk0 = 0.f, rk1 = 0.f, rk2 = 0.f, rk3 = 0.f;
    float rg0 = 0.f, rg1 = 0.f, rg2 = 0.f, rg3 = 0.f, rv = 0.f, rb = 0.f;
#define KDA_FETCH(tt)                                                                                        do {                                                                                                         const size_t base_ = ((size_t) (tt) * H + h) * DH + lane;                                                rq0 = q[base_]; rq1 = q[base_ + 32]; rq2 = q[base_ + 64]; rq3 = q[base_ + 96];                           rk0 = k[base_]; rk1 = k[base_ + 32]; rk2 = k[base_ + 64]; rk3 = k[base_ + 96];                           rg0 = g[base_]; rg1 = g[base_ + 32]; rg2 = g[base_ + 64]; rg3 = g[base_ + 96];                           rv = v[((size_t) (tt) * H + h) * DH + c];                                                                rb = beta[(size_t) (tt) * H + h];                                                                    } while (0)
    if (T > 0) KDA_FETCH(0);
    for (int t = 0; t < T; ++t) {
        const int b = t & 1;
        qs[b][lane] = rq0; qs[b][lane + 32] = rq1; qs[b][lane + 64] = rq2; qs[b][lane + 96] = rq3;
        ks[b][lane] = rk0; ks[b][lane + 32] = rk1; ks[b][lane + 64] = rk2; ks[b][lane + 96] = rk3;
        eg[b][lane] = expf(rg0); eg[b][lane + 32] = expf(rg1); eg[b][lane + 64] = expf(rg2); eg[b][lane + 96] = expf(rg3);
        const float vt = rv, bt = rb;
        __syncwarp();
        if (t + 1 < T) KDA_FETCH(t + 1);   // in flight while this token computes
        const float* egb = eg[b] + r0;
        const float* ksb = ks[b] + r0;
        const float* qsb = qs[b] + r0;
        float kv0 = 0.f, kv1 = 0.f;
#pragma unroll
        for (int i = 0; i < R; i += 2) {   // decay this half of the column, then its part of kv = S^T k
            st[i] *= egb[i];
            st[i + 1] *= egb[i + 1];
            kv0 += st[i] * ksb[i];
            kv1 += st[i + 1] * ksb[i + 1];
        }
        float kv = kv0 + kv1;
        kv += __shfl_xor_sync(0xffffffffu, kv, 16);
        const float delta = (vt - kv) * bt;
        float a0 = 0.f, a1 = 0.f;
#pragma unroll
        for (int i = 0; i < R; i += 2) {
            st[i] += ksb[i] * delta;
            st[i + 1] += ksb[i + 1] * delta;
            a0 += st[i] * qsb[i];
            a1 += st[i + 1] * qsb[i + 1];
        }
        float a = a0 + a1;
        a += __shfl_xor_sync(0xffffffffu, a, 16);
        if (lane < 16) o[((size_t) t * H + h) * DH + c] = a;
    }
#pragma unroll
    for (int i = 0; i < R; ++i) Sg[(size_t) (r0 + i) * DH + c] = st[i];
#undef KDA_FETCH
}

__global__ void kda_out_norm_rows_kernel(float* __restrict__ o, const bf16* __restrict__ w, const float* __restrict__ gate,
                                         float eps, int H, int dh) {
    const int h = blockIdx.x, t = blockIdx.y, i = threadIdx.x;
    const size_t j = ((size_t) t * H + h) * dh + i;
    const float x = o[j];
    const float inv = rsqrtf(block_sum(x * x) / (float) dh + eps);
    o[j] = x * inv * bf(w[i]) / (1.f + expf(-gate[j]));
}

__global__ void mla_attend_rows_kernel(const float* __restrict__ qa, const float* __restrict__ lat,
                                       const int32_t* __restrict__ sel, const int32_t* __restrict__ cnt, int sel_ld,
                                       int pos0, float scale, float* __restrict__ ctx, int H, int R) {
    extern __shared__ float p[];
    __shared__ float sum;
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const int n_sel = sel ? cnt[t] : pos0 + t + 1;
    const int32_t* st = sel ? sel + (size_t) t * sel_ld : nullptr;
    const float* qh = qa + ((size_t) t * H + h) * R;
    for (int j = wid; j < n_sel; j += nw) {
        const float* lj = lat + (size_t) (st ? st[j] : j) * R;
        float acc = 0.f;
        for (int c = lane; c < R; c += 32) acc += qh[c] * lj[c];
        acc = warp_sum(acc);
        if (lane == 0) p[j] = acc * scale;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = -FLT_MAX;
        for (int j = 0; j < n_sel; ++j) m = fmaxf(m, p[j]);
        float s = 0.f;
        for (int j = 0; j < n_sel; ++j) s += (p[j] = expf(p[j] - m));
        sum = s;
    }
    __syncthreads();
    float* ch = ctx + ((size_t) t * H + h) * R;
    for (int c = threadIdx.x; c < R; c += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < n_sel; ++j) acc += p[j] * lat[(size_t) (st ? st[j] : j) * R + c];
        ch[c] = acc / sum;
    }
}

__global__ void idx_pool_rows_kernel(const float* __restrict__ key, const float* __restrict__ gate,
                                     const bf16* __restrict__ ape, float* __restrict__ pooled, int p0, int kpool, int dim,
                                     int out_p0) {
    const int p = p0 + blockIdx.x, d = threadIdx.x;
    if (d >= dim) return;
    const float* kr = key + (size_t) p * kpool * dim;
    const float* gr = gate + (size_t) p * kpool * dim;
    float m = -FLT_MAX;
    for (int j = 0; j < kpool; ++j) m = fmaxf(m, gr[(size_t) j * dim + d] + bf(ape[(size_t) j * dim + d]));
    float s = 0.f, acc = 0.f;
    for (int j = 0; j < kpool; ++j) {
        const float e = expf(gr[(size_t) j * dim + d] + bf(ape[(size_t) j * dim + d]) - m);
        s += e;
        acc += e * kr[(size_t) j * dim + d];
    }
    pooled[(size_t) (out_p0 + blockIdx.x) * dim + d] = acc / s;
}

__global__ void idx_scores_rows_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                       const float* __restrict__ pooled, float* __restrict__ score, int n_pool, int H,
                                       int dim, int pos0, int kpool) {
    const int t = blockIdx.y, p = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (p >= n_pool) return;
    float* sr = score + (size_t) t * n_pool;
    if (p * kpool + kpool - 1 > pos0 + t) {   // the pool's last token is after this query
        if (lane == 0) sr[p] = -FLT_MAX;
        return;
    }
    const float* pk = pooled + (size_t) p * dim;
    const float* qt = q + (size_t) t * H * dim;
    const float* wt = w + (size_t) t * H;
    const float sc = rsqrtf((float) dim);
    float total = 0.f;
    for (int h = 0; h < H; ++h) {
        float acc = 0.f;
        for (int d = lane; d < dim; d += 32) acc += qt[(size_t) h * dim + d] * pk[d];
        acc = warp_sum(acc);
        total += wt[h] * fmaxf(acc * sc, 0.f);
    }
    if (lane == 0) sr[p] = total * rsqrtf((float) H);
}

__global__ void route_rows_kernel(const float* __restrict__ logits_all, const float* __restrict__ bias, int n, int k,
                                  float scaling, int32_t* __restrict__ ids_all, float* __restrict__ wts_all) {
    if (threadIdx.x != 0) return;
    const int t = blockIdx.x;
    const float* logits = logits_all + (size_t) t * n;
    float chosen[64];
    int idx[64];
    for (int j = 0; j < k; ++j) { chosen[j] = -FLT_MAX; idx[j] = -1; }
    for (int e = 0; e < n; ++e) {
        const float sc = 1.f / (1.f + expf(-logits[e])) + bias[e];
        int pos = k;
        while (pos > 0 && sc > chosen[pos - 1]) --pos;
        if (pos < k) {
            for (int j = k - 1; j > pos; --j) { chosen[j] = chosen[j - 1]; idx[j] = idx[j - 1]; }
            chosen[pos] = sc;
            idx[pos] = e;
        }
    }
    float sum = 0.f;
    for (int j = 0; j < k; ++j) sum += 1.f / (1.f + expf(-logits[idx[j]]));
    for (int j = 0; j < k; ++j) {
        ids_all[(size_t) t * k + j] = idx[j];
        wts_all[(size_t) t * k + j] = (1.f / (1.f + expf(-logits[idx[j]]))) / (sum + 1e-20f) * scaling;
    }
}

__global__ void combine_rows_t_kernel(const float* __restrict__ rows, const float* __restrict__ w, int k,
                                      float* __restrict__ y, int n) {
    const int t = blockIdx.y, i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = y[(size_t) t * n + i];
    for (int j = 0; j < k; ++j) acc += w[(size_t) t * k + j] * rows[((size_t) t * k + j) * n + i];
    y[(size_t) t * n + i] = acc;
}

}  // namespace

void layernorm_rows(const float* x, const bf16* w, const bf16* b, float eps, float* out, int n, int nrows, cudaStream_t s) {
    layernorm_rows_kernel<<<nrows, 128, 0, s>>>(x, w, b, eps, out, n);
    check("layernorm_rows");
}

void hc_pre_rows(const float* streams, const float* mix, const bf16* base, const bf16* scale, int n, float rms_eps,
                 float hc_eps, int iters, float* x, float* post, float* comb, int T, cudaStream_t s) {
    hc_pre_rows_kernel<<<T, 512, 0, s>>>(streams, mix, base, scale, n, rms_eps, hc_eps, iters, x, post, comb);
    check("hc_pre_rows");
}

void hc_post_rows(const float* y, const float* streams_in, const float* post, const float* comb, float* streams_out,
                  int n, int T, cudaStream_t s) {
    hc_post_rows_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(y, streams_in, post, comb,
                                                                                         streams_out, n);
    check("hc_post_rows");
}

static float* g_conv_tmp = nullptr;
static size_t g_conv_cap = 0;

void conv_silu_reserve(size_t n) {   // the scratch up front, before the expert tier sizes itself to the free VRAM
    if (n <= g_conv_cap) return;
    if (g_conv_tmp) cudaFree(g_conv_tmp);
    if (cudaMalloc(&g_conv_tmp, n * sizeof(float)) != cudaSuccess) { std::fprintf(stderr, "conv_silu_reserve: no memory\n"); std::exit(1); }
    g_conv_cap = n;
}

void conv_silu_release() {
    if (g_conv_tmp) cudaFree(g_conv_tmp);
    g_conv_tmp = nullptr;
    g_conv_cap = 0;
}

void conv_silu_seq(const float* x, const float* w, float* state, float* out, int C, int T, cudaStream_t s) {
    // out may alias x: the outputs go to a scratch first, then the state (from x) is updated, then copied back
    float*& tmp = g_conv_tmp;
    size_t& cap = g_conv_cap;
    const size_t n = (size_t) C * T;
    if (n > cap) {
        if (tmp) cudaFree(tmp);
        if (cudaMalloc(&tmp, n * sizeof(float)) != cudaSuccess) { std::fprintf(stderr, "conv_silu_seq: no scratch\n"); std::exit(1); }
        cap = n;
    }
    conv_silu_par_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(x, w, state, tmp, C, T);
    conv_state_kernel<<<(unsigned) ((C + 255) / 256), 256, 0, s>>>(x, state, C, T);
    cudaMemcpyAsync(out, tmp, n * sizeof(float), cudaMemcpyDeviceToDevice, s);
    check("conv_silu_seq");
}

void kda_prep_rows(float* q, float* k, float* gf, const float* dt_bias, const float* A_log, float lb, float* b, int H,
                   int dh, int T, cudaStream_t s) {
    kda_prep_rows_kernel<<<dim3((unsigned) H, (unsigned) T), dh, 0, s>>>(q, k, gf, dt_bias, A_log, lb, b, H, dh);
    check("kda_prep_rows");
}

void kda_scan(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta, float* o, int H,
              int dh, int T, cudaStream_t s) {
    if (dh != 128) { std::fprintf(stderr, "kda_scan: head dim %d (128 only)\n", dh); std::exit(1); }
    kda_scan_kernel<128><<<dim3((unsigned) H, 128 / 16), 32, 0, s>>>(S, q, k, v, g, beta, o, H, T);
    check("kda_scan");
}

void kda_out_norm_rows(float* o, const bf16* w, const float* gate, float eps, int H, int dh, int T, cudaStream_t s) {
    kda_out_norm_rows_kernel<<<dim3((unsigned) H, (unsigned) T), dh, 0, s>>>(o, w, gate, eps, H, dh);
    check("kda_out_norm_rows");
}

void mla_attend_rows(const float* qa, const float* lat, const int32_t* sel, const int32_t* cnt, int sel_ld, int pos0,
                     int max_sel, float scale, float* ctx, int H, int R, int T, cudaStream_t s) {
    const size_t shm = (size_t) max_sel * sizeof(float);
    static size_t attr = 0;
    if (shm > 48 * 1024 && shm > attr) {
        cudaFuncSetAttribute(mla_attend_rows_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shm);
        attr = shm;
    }
    mla_attend_rows_kernel<<<dim3((unsigned) H, (unsigned) T), 256, shm, s>>>(qa, lat, sel, cnt, sel_ld, pos0, scale,
                                                                               ctx, H, R);
    check("mla_attend_rows");
}

void idx_pool_rows(const float* key, const float* gate, const bf16* ape, float* pooled, int p0, int np, int kpool, int dim,
                   cudaStream_t s, int out_p0) {
    if (np <= 0) return;
    idx_pool_rows_kernel<<<np, dim, 0, s>>>(key, gate, ape, pooled, p0, kpool, dim, out_p0 < 0 ? p0 : out_p0);
    check("idx_pool_rows");
}

void idx_scores_rows(const float* q, const float* w, const float* pooled, float* score, int n_pool, int H, int dim,
                     int pos0, int kpool, int T, cudaStream_t s) {
    idx_scores_rows_kernel<<<dim3((unsigned) ((n_pool + 7) / 8), (unsigned) T), 256, 0, s>>>(q, w, pooled, score, n_pool,
                                                                                             H, dim, pos0, kpool);
    check("idx_scores_rows");
}

void route_rows(const float* logits, const float* bias, int n_expert, int k, float scaling, int32_t* ids, float* wts,
                int T, cudaStream_t s) {
    route_rows_kernel<<<T, 32, 0, s>>>(logits, bias, n_expert, k, scaling, ids, wts);
    check("route_rows");
}

void combine_rows_t(const float* rows, const float* w, int k, float* y, int n, int T, cudaStream_t s) {
    combine_rows_t_kernel<<<dim3((unsigned) ((n + 255) / 256), (unsigned) T), 256, 0, s>>>(rows, w, k, y, n);
    check("combine_rows_t");
}

}  // namespace glm

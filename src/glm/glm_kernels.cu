// src/glm/glm_kernels.cu - see glm_kernels.cuh.  M1: correct first; FP32 math throughout.
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

// block-wide sum (blockDim.x a multiple of 32, <= 1024)
__device__ float block_sum(float v) {
    __shared__ float part[32];
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) part[wid] = v;
    __syncthreads();
    float r = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0.f;
    if (wid == 0) r = warp_sum(r);
    __shared__ float total;
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

// ---------------------------------------------------------------- GEMV
template <int NT>
__global__ void __launch_bounds__(256) gemv_bf16_kernel(const bf16* __restrict__ W, const float* __restrict__ x,
                                                        float* __restrict__ y, int rows, int cols, int x_ld, int y_ld) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint4* wr = (const uint4*) (W + (size_t) row * cols);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int c8 = lane; c8 < cols / 8; c8 += 32) {
        const uint4 u = wr[c8];
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
        float wf[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            wf[2 * j] = __uint_as_float(w4[j] << 16);
            wf[2 * j + 1] = __uint_as_float(w4[j] & 0xffff0000u);
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4* xv = (const float4*) (x + (size_t) t * x_ld + (size_t) c8 * 8);
            const float4 a = xv[0], b = xv[1];
            acc[t] += wf[0] * a.x + wf[1] * a.y + wf[2] * a.z + wf[3] * a.w + wf[4] * b.x + wf[5] * b.y + wf[6] * b.z +
                      wf[7] * b.w;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float s = warp_sum(acc[t]);
        if (lane == 0) y[(size_t) t * y_ld + row] = s;
    }
}

__global__ void __launch_bounds__(256) gemv_f32_kernel(const float* __restrict__ W, const float* __restrict__ x,
                                                       float* __restrict__ y, int rows, int cols, int nt) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const float4* wr = (const float4*) (W + (size_t) row * cols);
    for (int t = 0; t < nt; ++t) {
        const float4* xv = (const float4*) (x + (size_t) t * cols);
        float acc = 0.f;
        for (int c4 = lane; c4 < cols / 4; c4 += 32) {
            const float4 w = wr[c4], a = xv[c4];
            acc += w.x * a.x + w.y * a.y + w.z * a.z + w.w * a.w;
        }
        acc = warp_sum(acc);
        if (lane == 0) y[(size_t) t * rows + row] = acc;
    }
}

// ---------------------------------------------------------------- norms
__global__ void rmsnorm_kernel(const float* __restrict__ x, const bf16* __restrict__ w, float eps, float* __restrict__ out,
                               int n) {
    const float* xr = x + (size_t) blockIdx.x * n;
    float* o = out + (size_t) blockIdx.x * n;
    float ss = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    const float inv = rsqrtf(block_sum(ss) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) o[i] = xr[i] * inv * (w ? bf(w[i]) : 1.f);
}

__global__ void layernorm_kernel(const float* __restrict__ x, const bf16* __restrict__ w, const bf16* __restrict__ b,
                                 float eps, float* __restrict__ out, int n) {
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i];
    const float mean = block_sum(s) / (float) n;
    float v = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) v += (x[i] - mean) * (x[i] - mean);
    const float inv = rsqrtf(block_sum(v) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = (x[i] - mean) * inv * bf(w[i]) + bf(b[i]);
}

// ---------------------------------------------------------------- mHC
__global__ void hc_pre_kernel(const float* __restrict__ streams, const float* __restrict__ mix,
                              const bf16* __restrict__ base, const bf16* __restrict__ scale, int n, float rms_eps,
                              float hc_eps, int iters, float* __restrict__ x, float* __restrict__ post_out,
                              float* __restrict__ comb_out) {
    __shared__ float pre[4], post[4], comb[16], inv_rms;
    float ss = 0.f;
    for (int i = threadIdx.x; i < 4 * n; i += blockDim.x) ss += streams[i] * streams[i];
    ss = block_sum(ss);
    if (threadIdx.x == 0) {
        inv_rms = rsqrtf(ss / (float) (4 * n) + rms_eps);
        const float s0 = bf(scale[0]), s1 = bf(scale[1]), s2 = bf(scale[2]);
        for (int j = 0; j < 4; ++j) {
            pre[j] = 1.f / (1.f + expf(-(mix[j] * inv_rms * s0 + bf(base[j])))) + hc_eps;
            post[j] = 2.f / (1.f + expf(-(mix[4 + j] * inv_rms * s1 + bf(base[4 + j]))));
        }
        // comb: softmax over each row (the last index), + eps, then columns, then (iters-1) x (rows, columns)
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
        auto cols = [&]() {
            for (int c = 0; c < 4; ++c) {
                float sum = 0.f;
                for (int r = 0; r < 4; ++r) sum += comb[4 * r + c];
                for (int r = 0; r < 4; ++r) comb[4 * r + c] /= sum + hc_eps;
            }
        };
        auto rows = [&]() {
            for (int r = 0; r < 4; ++r) {
                float sum = 0.f;
                for (int c = 0; c < 4; ++c) sum += comb[4 * r + c];
                for (int c = 0; c < 4; ++c) comb[4 * r + c] /= sum + hc_eps;
            }
        };
        cols();
        for (int it = 1; it < iters; ++it) { rows(); cols(); }
        for (int j = 0; j < 4; ++j) post_out[j] = post[j];
        for (int j = 0; j < 16; ++j) comb_out[j] = comb[j];
    }
    __syncthreads();
    for (int d = threadIdx.x; d < n; d += blockDim.x)
        x[d] = pre[0] * streams[d] + pre[1] * streams[n + d] + pre[2] * streams[2 * n + d] + pre[3] * streams[3 * n + d];
}

__global__ void hc_post_kernel(const float* __restrict__ y, const float* streams_in, const float* __restrict__ post,
                               const float* __restrict__ comb, float* streams_out, int n) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n) return;
    const float r0 = streams_in[d], r1 = streams_in[n + d], r2 = streams_in[2 * n + d], r3 = streams_in[3 * n + d];
    const float yd = y[d];
    for (int j = 0; j < 4; ++j)
        streams_out[j * n + d] = post[j] * yd + comb[0 * 4 + j] * r0 + comb[1 * 4 + j] * r1 + comb[2 * 4 + j] * r2 +
                                 comb[3 * 4 + j] * r3;
}

__global__ void hc_mean_kernel(const float* __restrict__ streams, float* __restrict__ out, int n) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < n) out[d] = (streams[d] + streams[n + d] + streams[2 * n + d] + streams[3 * n + d]) * 0.25f;
}

// ---------------------------------------------------------------- KDA
__global__ void conv_silu_kernel(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ state,
                                 float* __restrict__ out, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float* st = state + (size_t) c * 3;
    const float* wc = w + (size_t) c * 4;
    const float v = wc[0] * st[0] + wc[1] * st[1] + wc[2] * st[2] + wc[3] * x[c];
    st[0] = st[1];
    st[1] = st[2];
    st[2] = x[c];
    out[c] = v / (1.f + expf(-v));
}

__global__ void kda_prep_kernel(float* __restrict__ q, float* __restrict__ k, float* __restrict__ gf,
                                const float* __restrict__ dt_bias, const float* __restrict__ A_log, float lb,
                                float* __restrict__ b, int dh) {
    const int h = blockIdx.x, i = threadIdx.x;   // blockDim = dh
    float* qh = q + (size_t) h * dh;
    float* kh = k + (size_t) h * dh;
    const float qn = block_sum(qh[i] * qh[i]), kn = block_sum(kh[i] * kh[i]);
    qh[i] = qh[i] / sqrtf(qn + 1e-6f) * rsqrtf((float) dh);
    kh[i] = kh[i] / sqrtf(kn + 1e-6f);
    const size_t j = (size_t) h * dh + i;
    gf[j] = lb / (1.f + expf(-(expf(A_log[h]) * (gf[j] + dt_bias[j]))));
    if (i == 0) b[h] = 1.f / (1.f + expf(-b[h]));
}

__global__ void kda_step_kernel(float* __restrict__ S, const float* __restrict__ q, const float* __restrict__ k,
                                const float* __restrict__ v, const float* __restrict__ g, const float* __restrict__ beta,
                                float* __restrict__ o, int dh) {
    extern __shared__ float sm[];
    float *qs = sm, *ks = sm + dh, *eg = sm + 2 * dh;
    const int h = blockIdx.x, vi = threadIdx.x;   // one thread per value column
    const size_t base = (size_t) h * dh;
    qs[vi] = q[base + vi];
    ks[vi] = k[base + vi];
    eg[vi] = expf(g[base + vi]);
    __syncthreads();
    float* Sh = S + (size_t) h * dh * dh;
    float kv = 0.f;
    for (int ki = 0; ki < dh; ++ki) kv += Sh[(size_t) ki * dh + vi] * eg[ki] * ks[ki];
    const float delta = (v[base + vi] - kv) * beta[h];
    float acc = 0.f;
    for (int ki = 0; ki < dh; ++ki) {
        const float s = Sh[(size_t) ki * dh + vi] * eg[ki] + ks[ki] * delta;
        Sh[(size_t) ki * dh + vi] = s;
        acc += s * qs[ki];
    }
    o[base + vi] = acc;
}

__global__ void kda_out_norm_kernel(float* __restrict__ o, const bf16* __restrict__ w, const float* __restrict__ gate,
                                    float eps, int dh) {
    const int h = blockIdx.x, i = threadIdx.x;
    const size_t j = (size_t) h * dh + i;
    const float x = o[j];
    const float inv = rsqrtf(block_sum(x * x) / (float) dh + eps);
    o[j] = x * inv * bf(w[i]) / (1.f + expf(-gate[j]));
}

// ---------------------------------------------------------------- MLA
__global__ void mla_absorb_kernel(const float* __restrict__ q, const bf16* __restrict__ kvb, float* __restrict__ qa,
                                  int dk, int dv, int R) {
    const int h = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= R) return;
    const bf16* Wh = kvb + (size_t) h * (dk + dv) * R;
    const float* qh = q + (size_t) h * dk;
    float acc = 0.f;
    for (int d = 0; d < dk; ++d) acc += qh[d] * bf(Wh[(size_t) d * R + c]);
    qa[(size_t) h * R + c] = acc;
}

__global__ void mla_attend_kernel(const float* __restrict__ qa, const float* __restrict__ lat,
                                  const int32_t* __restrict__ sel, int n_sel, float scale, float* __restrict__ ctx, int R) {
    extern __shared__ float p[];                 // n_sel scores
    const int h = blockIdx.x, lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5;
    const float* qh = qa + (size_t) h * R;
    for (int j = wid; j < n_sel; j += nw) {
        const float* lj = lat + (size_t) sel[j] * R;
        float acc = 0.f;
        for (int c = lane; c < R; c += 32) acc += qh[c] * lj[c];
        acc = warp_sum(acc);
        if (lane == 0) p[j] = acc * scale;
    }
    __syncthreads();
    __shared__ float mx, sum;
    if (threadIdx.x == 0) {
        float m = -FLT_MAX;
        for (int j = 0; j < n_sel; ++j) m = fmaxf(m, p[j]);
        float s = 0.f;
        for (int j = 0; j < n_sel; ++j) s += (p[j] = expf(p[j] - m));
        mx = m;
        sum = s;
    }
    __syncthreads();
    for (int c = threadIdx.x; c < R; c += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < n_sel; ++j) acc += p[j] * lat[(size_t) sel[j] * R + c];
        ctx[(size_t) h * R + c] = acc / sum;
    }
    (void) mx;
}

__global__ void mla_value_kernel(const float* __restrict__ ctx, const bf16* __restrict__ kvb, float* __restrict__ out,
                                 int dk, int dv, int R) {
    const int h = blockIdx.y, e = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (e >= dv) return;
    const bf16* wr = kvb + ((size_t) h * (dk + dv) + dk + e) * R;
    const float* ch = ctx + (size_t) h * R;
    float acc = 0.f;
    for (int c = lane; c < R; c += 32) acc += bf(wr[c]) * ch[c];
    acc = warp_sum(acc);
    if (lane == 0) out[(size_t) h * dv + e] = acc;
}

// ---------------------------------------------------------------- indexer
__global__ void idx_pool_kernel(const float* __restrict__ key, const float* __restrict__ gate,
                                const bf16* __restrict__ ape, float* __restrict__ pooled, int kpool, int dim) {
    const int d = threadIdx.x;
    if (d >= dim) return;
    float m = -FLT_MAX;
    for (int j = 0; j < kpool; ++j) m = fmaxf(m, gate[(size_t) j * dim + d] + bf(ape[(size_t) j * dim + d]));
    float s = 0.f, acc = 0.f;
    for (int j = 0; j < kpool; ++j) {
        const float e = expf(gate[(size_t) j * dim + d] + bf(ape[(size_t) j * dim + d]) - m);
        s += e;
        acc += e * key[(size_t) j * dim + d];
    }
    pooled[d] = acc / s;
}

__global__ void idx_scores_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                  const float* __restrict__ pooled, float* __restrict__ score, int n_pool, int H,
                                  int dim) {
    const int p = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (p >= n_pool) return;
    const float* pk = pooled + (size_t) p * dim;
    const float sc = rsqrtf((float) dim);
    float total = 0.f;
    for (int h = 0; h < H; ++h) {
        const float* qh = q + (size_t) h * dim;
        float acc = 0.f;
        for (int d = lane; d < dim; d += 32) acc += qh[d] * pk[d];
        acc = warp_sum(acc);
        total += w[h] * fmaxf(acc * sc, 0.f);
    }
    if (lane == 0) score[p] = total * rsqrtf((float) H);   // weights_proj(x) * H^-0.5
}

// ---------------------------------------------------------------- MoE bits
__global__ void route_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int n, int k,
                             float scaling, int32_t* __restrict__ ids, float* __restrict__ wts) {
    if (threadIdx.x != 0) return;
    float chosen[64];
    int idx[64];
    for (int j = 0; j < k; ++j) { chosen[j] = -FLT_MAX; idx[j] = -1; }
    for (int e = 0; e < n; ++e) {
        const float sc = 1.f / (1.f + expf(-logits[e])) + bias[e];
        int pos = k;
        while (pos > 0 && sc > chosen[pos - 1]) --pos;   // descending; ties keep the lower id first
        if (pos < k) {
            for (int j = k - 1; j > pos; --j) { chosen[j] = chosen[j - 1]; idx[j] = idx[j - 1]; }
            chosen[pos] = sc;
            idx[pos] = e;
        }
    }
    float sum = 0.f;
    for (int j = 0; j < k; ++j) sum += 1.f / (1.f + expf(-logits[idx[j]]));
    for (int j = 0; j < k; ++j) {
        ids[j] = idx[j];
        wts[j] = (1.f / (1.f + expf(-logits[idx[j]]))) / (sum + 1e-20f) * scaling;
    }
}

__global__ void swiglu_clamp_kernel(const float* __restrict__ g, const float* __restrict__ u, float* __restrict__ h,
                                    int n, float lim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gg = fminf(g[i], lim), uu = fminf(fmaxf(u[i], -lim), lim);
    h[i] = gg / (1.f + expf(-gg)) * uu;
}

__global__ void combine_kernel(const float* __restrict__ rows, const float* __restrict__ w, int k, float* __restrict__ y,
                               int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = y[i];
    for (int e = 0; e < k; ++e) acc += w[e] * rows[(size_t) e * n + i];
    y[i] = acc;
}

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a[i] + b[i];
}

__device__ __forceinline__ float e4m3f(uint8_t b) {
    const int e = (b >> 3) & 0xF, m = b & 7;
    const float v = e == 0 ? (float) m / 8.f * 0.015625f : (1.f + (float) m / 8.f) * exp2f((float) (e - 7));
    return (b & 0x80) ? -v : v;
}

__global__ void nvfp4_to_f32_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc, float s2,
                                    float* __restrict__ out, int rows, int cols) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;   // one byte = two values
    if (i >= (long long) rows * cols / 2) return;
    const float E2M1[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
    const long long r = i / (cols / 2), cb = i % (cols / 2), c = 2 * cb;
    const float s = e4m3f(sc[r * (cols / 16) + c / 16]) * s2;
    const uint8_t b = w[i];
    out[r * cols + c] = E2M1[b & 0xF] * s;
    out[r * cols + c + 1] = E2M1[b >> 4] * s;
}

__global__ void bf16_to_f32_kernel(const bf16* in, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = bf(in[i]);
}

}  // namespace

void gemv_bf16(const bf16* W, const float* x, float* y, int rows, int cols, int nt, cudaStream_t s, int x_ld, int y_ld) {
    if (cols % 8) { std::fprintf(stderr, "gemv_bf16: cols %d not a multiple of 8\n", cols); std::exit(1); }
    x_ld = x_ld ? x_ld : cols;
    y_ld = y_ld ? y_ld : rows;
    const dim3 g((unsigned) ((rows + 7) / 8));
    switch (nt) {
        case 1: gemv_bf16_kernel<1><<<g, 256, 0, s>>>(W, x, y, rows, cols, x_ld, y_ld); break;
        case 2: gemv_bf16_kernel<2><<<g, 256, 0, s>>>(W, x, y, rows, cols, x_ld, y_ld); break;
        case 4: gemv_bf16_kernel<4><<<g, 256, 0, s>>>(W, x, y, rows, cols, x_ld, y_ld); break;
        case 8: gemv_bf16_kernel<8><<<g, 256, 0, s>>>(W, x, y, rows, cols, x_ld, y_ld); break;
        default:
            for (int t = 0; t < nt; ++t)
                gemv_bf16_kernel<1><<<g, 256, 0, s>>>(W, x + (size_t) t * x_ld, y + (size_t) t * y_ld, rows, cols, x_ld, y_ld);
    }
    check("gemv_bf16");
}

void gemv_f32(const float* W, const float* x, float* y, int rows, int cols, int nt, cudaStream_t s) {
    gemv_f32_kernel<<<(unsigned) ((rows + 7) / 8), 256, 0, s>>>(W, x, y, rows, cols, nt);
    check("gemv_f32");
}

void rmsnorm(const float* x, const bf16* w, float eps, float* out, int n, int nrows, cudaStream_t s) {
    rmsnorm_kernel<<<nrows, 256, 0, s>>>(x, w, eps, out, n);
    check("rmsnorm");
}

void layernorm(const float* x, const bf16* w, const bf16* b, float eps, float* out, int n, cudaStream_t s) {
    layernorm_kernel<<<1, 128, 0, s>>>(x, w, b, eps, out, n);
    check("layernorm");
}

void hc_pre_finish(const float* streams, const float* mix, const bf16* base, const bf16* scale, int n, float rms_eps,
                   float hc_eps, int iters, float* x, float* post, float* comb, cudaStream_t s) {
    hc_pre_kernel<<<1, 1024, 0, s>>>(streams, mix, base, scale, n, rms_eps, hc_eps, iters, x, post, comb);
    check("hc_pre");
}

void hc_post(const float* y, const float* streams_in, const float* post, const float* comb, float* streams_out, int n,
             cudaStream_t s) {
    hc_post_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(y, streams_in, post, comb, streams_out, n);
    check("hc_post");
}

void hc_mean(const float* streams, float* out, int n, cudaStream_t s) {
    hc_mean_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(streams, out, n);
    check("hc_mean");
}

void conv_silu_step(const float* x, const float* w, float* state, float* out, int C, cudaStream_t s) {
    conv_silu_kernel<<<(unsigned) ((C + 255) / 256), 256, 0, s>>>(x, w, state, out, C);
    check("conv_silu");
}

void kda_prep(float* q, float* k, float* gf, const float* dt_bias, const float* A_log, float lb, float* b, int H, int dh,
              cudaStream_t s) {
    kda_prep_kernel<<<H, dh, 0, s>>>(q, k, gf, dt_bias, A_log, lb, b, dh);
    check("kda_prep");
}

void kda_step(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta, float* o,
              int H, int dh, cudaStream_t s) {
    kda_step_kernel<<<H, dh, 3 * dh * sizeof(float), s>>>(S, q, k, v, g, beta, o, dh);
    check("kda_step");
}

void kda_out_norm(float* o, const bf16* w, const float* gate, float eps, int H, int dh, cudaStream_t s) {
    kda_out_norm_kernel<<<H, dh, 0, s>>>(o, w, gate, eps, dh);
    check("kda_out_norm");
}

void mla_absorb_q(const float* q, const bf16* kvb, float* qa, int H, int dk, int dv, int R, cudaStream_t s) {
    mla_absorb_kernel<<<dim3((unsigned) ((R + 127) / 128), (unsigned) H), 128, 0, s>>>(q, kvb, qa, dk, dv, R);
    check("mla_absorb");
}

void mla_attend(const float* qa, const float* lat, const int32_t* sel, int n_sel, float scale, float* ctx, int H, int R,
                cudaStream_t s) {
    mla_attend_kernel<<<H, 256, (size_t) n_sel * sizeof(float), s>>>(qa, lat, sel, n_sel, scale, ctx, R);
    check("mla_attend");
}

void mla_value(const float* ctx, const bf16* kvb, float* out, int H, int dk, int dv, int R, cudaStream_t s) {
    mla_value_kernel<<<dim3((unsigned) ((dv + 7) / 8), (unsigned) H), 256, 0, s>>>(ctx, kvb, out, dk, dv, R);
    check("mla_value");
}

void idx_pool(const float* key, const float* gate, const bf16* ape, float* pooled, int kpool, int dim, cudaStream_t s) {
    idx_pool_kernel<<<1, dim, 0, s>>>(key, gate, ape, pooled, kpool, dim);
    check("idx_pool");
}

void idx_scores(const float* q, const float* w, const float* pooled, float* score, int n_pool, int H, int dim,
                cudaStream_t s) {
    idx_scores_kernel<<<(unsigned) ((n_pool + 7) / 8), 256, 0, s>>>(q, w, pooled, score, n_pool, H, dim);
    check("idx_scores");
}

void route_topk(const float* logits, const float* bias, int n_expert, int k, float scaling, int32_t* ids, float* wts,
                cudaStream_t s) {
    route_kernel<<<1, 32, 0, s>>>(logits, bias, n_expert, k, scaling, ids, wts);
    check("route");
}

void swiglu_clamp(const float* g, const float* u, float* h, int n, float lim, cudaStream_t s) {
    swiglu_clamp_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(g, u, h, n, lim);
    check("swiglu_clamp");
}

void combine_rows(const float* rows, const float* w, int k, float* y, int n, cudaStream_t s) {
    combine_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(rows, w, k, y, n);
    check("combine");
}

void add(const float* a, const float* b, float* out, int n, cudaStream_t s) {
    add_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(a, b, out, n);
    check("add");
}

void nvfp4_to_f32(const uint8_t* w, const uint8_t* sc, float s2, float* out, int rows, int cols, cudaStream_t s) {
    const long long n = (long long) rows * cols / 2;
    nvfp4_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(w, sc, s2, out, rows, cols);
    check("nvfp4_to_f32");
}

void bf16_to_f32(const bf16* in, float* out, int n, cudaStream_t s) {
    bf16_to_f32_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(in, out, n);
    check("bf16_to_f32");
}

}  // namespace glm

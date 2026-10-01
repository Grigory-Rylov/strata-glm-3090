// src/glm/glm_select.cu - the DSA indexer's top-k pool selection on the GPU (see idx_select_rows).
//
// A query at position pos sees vp = (pos + 1) / kpool complete pools. Up to `budget` of them it attends to every
// token; past that, to the `budget` pools with the highest indexer score plus its incomplete tail. With 262K of
// context that is 65K scores a query: a radix select on the scores' order-preserving bits (4 passes of 8 bits)
// finds the budget-th largest, and one ordered pass writes the chosen pools by index (ties at the threshold: the
// lower pools), so a run is deterministic and needs no host round trip.
#include "glm_kernels.cuh"

#include <cuda_fp16.h>
#include <mma.h>

#include <cfloat>

#include <cstdio>
#include <cstdlib>

namespace glm {
namespace {

__device__ __forceinline__ uint32_t okey(float f) {   // larger float -> larger key
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// exclusive prefix sum over the block, in thread order
__device__ int block_excl_scan(int v, int* tmp) {
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (int) (blockDim.x >> 5);
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) tmp[w] = x;
    __syncthreads();
    if (w == 0) {
        int s = lane < nw ? tmp[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(0xffffffffu, s, o);
            if (lane >= o) s += y;
        }
        tmp[lane] = s;
    }
    __syncthreads();
    const int r = x - v + (w > 0 ? tmp[w - 1] : 0);
    __syncthreads();
    return r;
}

__global__ void __launch_bounds__(1024) idx_select_kernel(const float* __restrict__ score, int ld, int pos0, int kpool,
                                                          int budget, int32_t* __restrict__ sel, int sel_ld,
                                                          int32_t* __restrict__ cnt) {
    const int t = blockIdx.x, pos = pos0 + t, vp = (pos + 1) / kpool, nt = (int) blockDim.x, tid = (int) threadIdx.x;
    int32_t* row = sel + (size_t) t * sel_ld;
    if (vp <= budget) {
        for (int q = tid; q <= pos; q += nt) row[q] = q;
        if (tid == 0) cnt[t] = pos + 1;
        return;
    }
    const float* sc = score + (size_t) t * ld;
    __shared__ int hist[256];
    __shared__ int tmp[32];
    __shared__ uint32_t s_prefix;
    __shared__ int s_remaining;
    uint32_t prefix = 0, mask = 0;
    int remaining = budget;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = tid; i < 256; i += nt) hist[i] = 0;
        __syncthreads();
        for (int i = tid; i < vp; i += nt) {
            const uint32_t k = okey(sc[i]);
            if ((k & mask) == prefix) atomicAdd(&hist[(k >> shift) & 255], 1);
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0, b = 255;
            for (; b > 0; --b) {
                if (acc + hist[b] >= remaining) break;
                acc += hist[b];
            }
            s_remaining = remaining - acc;
            s_prefix = prefix | ((uint32_t) b << shift);
        }
        __syncthreads();
        remaining = s_remaining;
        prefix = s_prefix;
        mask |= 255u << shift;
        __syncthreads();
    }
    // prefix is the budget-th largest key: every key above it, and the first `remaining` equal to it
    const int per = (vp + nt - 1) / nt, i0 = min(vp, tid * per), i1 = min(vp, i0 + per);
    int n_gt = 0, n_eq = 0;
    for (int i = i0; i < i1; ++i) {
        const uint32_t k = okey(sc[i]);
        n_gt += k > prefix;
        n_eq += k == prefix;
    }
    const int eq_before = block_excl_scan(n_eq, tmp);
    const int take_eq = max(0, min(n_eq, remaining - eq_before));
    int o = block_excl_scan(n_gt + take_eq, tmp), e_seen = 0;
    for (int i = i0; i < i1; ++i) {
        const uint32_t k = okey(sc[i]);
        if (k > prefix || (k == prefix && e_seen++ < take_eq)) {
            for (int j = 0; j < kpool; ++j) row[o * kpool + j] = i * kpool + j;
            ++o;
        }
    }
    const int base = budget * kpool, tail0 = vp * kpool;
    for (int q = tail0 + tid; q <= pos; q += nt) row[base + (q - tail0)] = q;
    if (tid == 0) cnt[t] = base + (pos + 1 - tail0);
}

// The indexer's scores on tensor cores: score[t][p] = H^-1/2 sum_h w[t][h] relu(q[t][h] . pooled[p] / sqrt(D)), D =
// 128, H = 32. A block takes 2 queries (64 head rows of Q, FP16 in shared memory) and walks a span of pools 64 at a
// time: S = Q P^T in WMMA fragments, then each (query, pool) sums its 32 heads. Pools whose last token comes after
// the query are -FLT_MAX. Quadratic in the context (a query sees every pool), so at 262K it matters.
constexpr int kD = 128, kQH = 32, kQT = 2, kPT = 64, kDl = kD + 8;
__global__ void __launch_bounds__(128) idx_scores_tc_kernel(const float* __restrict__ q, const float* __restrict__ w,
                                                            const float* __restrict__ pooled, float* __restrict__ score,
                                                            int n_pool, int pos0, int kpool, int T, int span) {
    using namespace nvcuda;
    extern __shared__ __align__(32) unsigned char sm[];
    __half* Qs = (__half*) sm;                                   // [kQT * kQH][kDl]
    __half* Ps = Qs + kQT * kQH * kDl;                            // [kPT][kDl]
    float* Ss = (float*) (Ps + kPT * kDl);                       // [kQT * kQH][kPT]
    const int t0 = blockIdx.y * kQT, p_begin = blockIdx.x * span, tid = threadIdx.x, warp = tid >> 5;
    const float sc = rsqrtf((float) kD), hs = rsqrtf((float) kQH);
    for (int i = tid; i < kQT * kQH * kD; i += 128) {
        const int r = i / kD, d = i % kD, t = t0 + r / kQH;
        Qs[r * kDl + d] = __float2half(t < T ? q[(size_t) t * kQH * kD + (size_t) (r % kQH) * kD + d] : 0.f);
    }
    for (int pb = p_begin; pb < min(n_pool, p_begin + span); pb += kPT) {
        __syncthreads();
        for (int i = tid; i < kPT * kD; i += 128) {
            const int r = i / kD, d = i % kD, pp = pb + r;
            Ps[r * kDl + d] = __float2half(pp < n_pool ? pooled[(size_t) pp * kD + d] : 0.f);
        }
        __syncthreads();
        // warp w: head rows 16w..16w+15 x all 64 pools
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
        for (int f = 0; f < 4; ++f) wmma::fill_fragment(acc[f], 0.f);
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
        for (int kk = 0; kk < kD; kk += 16) {
            wmma::load_matrix_sync(a, Qs + (warp * 16) * kDl + kk, kDl);
            for (int f = 0; f < 4; ++f) {
                wmma::load_matrix_sync(b, Ps + (f * 16) * kDl + kk, kDl);
                wmma::mma_sync(acc[f], a, b, acc[f]);
            }
        }
        for (int f = 0; f < 4; ++f) wmma::store_matrix_sync(Ss + (warp * 16) * kPT + f * 16, acc[f], kPT, wmma::mem_row_major);
        __syncthreads();
        {   // 128 threads = 2 queries x 64 pools
            const int qi = tid / kPT, pl = tid % kPT, t = t0 + qi, pp = pb + pl;
            if (t < T && pp < n_pool) {
                float total = -FLT_MAX;
                if (pp * kpool + kpool - 1 <= pos0 + t) {
                    total = 0.f;
                    const float* wt = w + (size_t) t * kQH;
                    for (int h = 0; h < kQH; ++h) total += wt[h] * fmaxf(Ss[(qi * kQH + h) * kPT + pl] * sc, 0.f);
                    total *= hs;
                }
                score[(size_t) t * n_pool + pp] = total;
            }
        }
    }
}

}  // namespace

void idx_scores_tc(const float* q, const float* w, const float* pooled, float* score, int n_pool, int pos0, int kpool, int T,
                   cudaStream_t s) {
    constexpr size_t smem = (size_t) (kQT * kQH + kPT) * kDl * 2 + (size_t) kQT * kQH * kPT * 4;
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(idx_scores_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem);
        attr = true;
    }
    const int span = 1024;
    idx_scores_tc_kernel<<<dim3((unsigned) ((n_pool + span - 1) / span), (unsigned) ((T + kQT - 1) / kQT)), 128, smem, s>>>(
        q, w, pooled, score, n_pool, pos0, kpool, T, span);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm idx_scores_tc: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void idx_select_rows(const float* score, int ld, int pos0, int kpool, int budget, int32_t* sel, int sel_ld, int32_t* cnt,
                     int T, cudaStream_t s) {
    idx_select_kernel<<<(unsigned) T, 1024, 0, s>>>(score, ld, pos0, kpool, budget, sel, sel_ld, cnt);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm idx_select_rows: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace glm

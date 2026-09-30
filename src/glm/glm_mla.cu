// src/glm/glm_mla.cu - MLA attention (absorbed, NoPE) on tensor cores: ctx[t][h] = softmax_j(qa[t][h] . lat[j] *
// scale) . lat[j] over query t's keys j (its DSA selection, or every position <= pos0 + t).
//
// All 64 heads share one latent row per key, so a block takes a query and 16 heads: each key tile (64 latent rows,
// FP16 in shared memory) serves 16 heads at once instead of being read by 64 separate blocks. Two passes over the
// keys: the first finds each head's max and exp-sum, the second accumulates normalized P . V in WMMA fragments
// (so no running rescale of the accumulators, whose element layout WMMA hides). Decode (one query) splits the keys
// over blocks (flash-decoding) and a second kernel merges the parts.
#include "glm_kernels.cuh"

#include <cuda_fp16.h>
#include <mma.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>

namespace glm {
namespace {

using namespace nvcuda;
constexpr int kR = 512, kHG = 16, kKT = 64, kLd = kR + 8, kThreads = 128;   // 4 warps
constexpr size_t kSmem = (size_t) kHG * kLd * 2 + (size_t) kKT * kLd * 2 + (size_t) kHG * kKT * 4 +
                         (size_t) kHG * (kKT + 8) * 2 + 2 * kHG * 4;

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm mla %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// grid (T, 64 / kHG, nsplit); part (nsplit > 1): per (t, split, head) its unnormalized O [512], max and sum
__global__ void __launch_bounds__(kThreads) mla_tc_kernel(const float* __restrict__ qa, const float* __restrict__ lat,
                                                          const int32_t* __restrict__ sel, const int32_t* __restrict__ cnt,
                                                          int sel_ld, int pos0, float scale, float* __restrict__ ctx,
                                                          float* __restrict__ part, int nsplit) {
    extern __shared__ __align__(16) unsigned char sm[];
    __half* Qs = (__half*) sm;                      // [kHG][kLd]
    __half* Ks = Qs + kHG * kLd;                           // [kKT][kLd]
    float* Ss = (float*) (Ks + kKT * kLd);                        // [kHG][kKT]
    __half* Ps = (__half*) (Ss + kHG * kKT);        // [kHG][kKT + 8]
    float* mh = (float*) (Ps + kHG * (kKT + 8));                  // [kHG]
    float* lh = mh + kHG;                                         // [kHG]
    const int t = blockIdx.x, g = blockIdx.y, sp = blockIdx.z, tid = threadIdx.x, warp = tid >> 5;
    const int n_all = sel ? cnt[t] : pos0 + t + 1;
    const int32_t* st = sel ? sel + (size_t) t * sel_ld : nullptr;
    const int per = (n_all + nsplit - 1) / nsplit, k0 = sp * per, k1 = min(n_all, k0 + per);
    // Q: 16 heads of this query in FP16 (11-bit significand: BF16's 8 cost 0.008 of KL on a long prompt), scaled
    // down by a power of two when its largest value would come near FP16's range; the scores get it back
    const float* q = qa + ((size_t) t * 64 + (size_t) g * kHG) * kR;
    __shared__ float qmax_w[kThreads / 32];
    float am = 0.f;
    for (int i = tid; i < kHG * kR; i += kThreads) am = fmaxf(am, fabsf(q[i]));
    for (int o = 16; o > 0; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
    if ((tid & 31) == 0) qmax_w[warp] = am;
    __syncthreads();
    am = fmaxf(fmaxf(qmax_w[0], qmax_w[1]), fmaxf(qmax_w[2], qmax_w[3]));
    const float qsc = am > 16384.f ? exp2f(ceilf(log2f(am / 16384.f))) : 1.f;
    scale *= qsc;
    for (int i = tid; i < kHG * kR; i += kThreads) Qs[(i / kR) * kLd + (i % kR)] = __float2half(q[i] / qsc);
    if (tid < kHG) { mh[tid] = -FLT_MAX; lh[tid] = 0.f; }
    auto load_tile = [&](int kb) {   // keys kb..kb+63 (zero rows past k1)
        for (int i = tid; i < kKT * (kR / 4); i += kThreads) {
            const int r = i / (kR / 4), c4 = i % (kR / 4), k = kb + r;
            float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
            if (k < k1) v = ((const float4*) (lat + (size_t) (st ? st[k] : k) * kR))[c4];
            __half2* d = (__half2*) (Ks + r * kLd + c4 * 4);
            d[0] = __floats2half2_rn(v.x, v.y);
            d[1] = __floats2half2_rn(v.z, v.w);
        }
    };
    auto scores = [&]() {   // Ss[h][k] = Q . K^T, warp w: keys 16w..16w+15
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
        wmma::fill_fragment(acc, 0.f);
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b;
        for (int kk = 0; kk < kR; kk += 16) {
            wmma::load_matrix_sync(a, Qs + kk, kLd);
            wmma::load_matrix_sync(b, Ks + (warp * 16) * kLd + kk, kLd);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(Ss + warp * 16, acc, kKT, wmma::mem_row_major);
    };
    __syncthreads();
    // pass 1: per head, the max and the exp-sum over this block's keys
    for (int kb = k0; kb < k1; kb += kKT) {
        load_tile(kb);
        __syncthreads();
        scores();
        __syncthreads();
        if (tid < kHG) {   // one thread per head (64 values)
            const int nk = min(kKT, k1 - kb);
            float m = mh[tid], l = lh[tid];
            for (int k = 0; k < nk; ++k) {
                const float v = Ss[tid * kKT + k] * scale;
                if (v > m) { l = l * __expf(m - v) + 1.f; m = v; }
                else l += __expf(v - m);
            }
            mh[tid] = m;
            lh[tid] = l;
        }
        __syncthreads();
    }
    // pass 2: O += P . V with P = exp(s - m) (normalized by l at the end, or by the merge)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> o[8];   // warp w: dims 128w..128w+127
    for (int f = 0; f < 8; ++f) wmma::fill_fragment(o[f], 0.f);
    for (int kb = k0; kb < k1; kb += kKT) {
        load_tile(kb);
        __syncthreads();
        scores();
        __syncthreads();
        const int nk = min(kKT, k1 - kb);
        for (int i = tid; i < kHG * kKT; i += kThreads) {
            const int h = i / kKT, k = i % kKT;
            const float p = k < nk ? __expf(Ss[i] * scale - mh[h]) : 0.f;
            Ps[h * (kKT + 8) + k] = __float2half(p);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b;
        for (int kk = 0; kk < kKT; kk += 16) {
            wmma::load_matrix_sync(a, Ps + kk, kKT + 8);
            for (int f = 0; f < 8; ++f) {
                wmma::load_matrix_sync(b, Ks + kk * kLd + warp * 128 + f * 16, kLd);
                wmma::mma_sync(o[f], a, b, o[f]);
            }
        }
        __syncthreads();
    }
    // out: normalized into ctx (one split), else the unnormalized part with its max and sum
    if (nsplit == 1) {
        float* c = ctx + ((size_t) t * 64 + (size_t) g * kHG) * kR;
        for (int f = 0; f < 8; ++f) wmma::store_matrix_sync(c + warp * 128 + f * 16, o[f], kR, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < kHG * kR; i += kThreads) {
            const float l = lh[i / kR];
            c[(size_t) (i / kR) * kR + (i % kR)] = l > 0.f ? c[(size_t) (i / kR) * kR + (i % kR)] / l : 0.f;
        }
    } else {
        // part: O [T][nsplit][64][512], then (max, sum) [T][nsplit][64][2]
        const size_t base = ((size_t) t * nsplit + sp) * 64 + (size_t) g * kHG;
        float* po = part + base * kR;
        for (int f = 0; f < 8; ++f) wmma::store_matrix_sync(po + warp * 128 + f * 16, o[f], kR, wmma::mem_row_major);
        float* ml = part + (size_t) gridDim.x * nsplit * 64 * kR + base * 2;
        if (tid < kHG) { ml[2 * tid] = mh[tid]; ml[2 * tid + 1] = lh[tid]; }
    }
}

// ctx[t][h] = sum_s O_s exp(m_s - M) / sum_s l_s exp(m_s - M)
__global__ void mla_merge_kernel(const float* __restrict__ part, float* __restrict__ ctx, int nsplit) {
    const int t = blockIdx.x, h = blockIdx.y;
    const float* ml = part + (size_t) gridDim.x * nsplit * 64 * kR;
    __shared__ float w[64];
    __shared__ float inv;
    if (threadIdx.x == 0) {
        float M = -FLT_MAX;
        for (int s = 0; s < nsplit; ++s) M = fmaxf(M, ml[(((size_t) t * nsplit + s) * 64 + h) * 2]);
        float L = 0.f;
        for (int s = 0; s < nsplit; ++s) {
            const float* q = ml + (((size_t) t * nsplit + s) * 64 + h) * 2;
            const float e = q[1] > 0.f ? __expf(q[0] - M) : 0.f;
            w[s] = e;
            L += q[1] * e;
        }
        inv = L > 0.f ? 1.f / L : 0.f;
    }
    __syncthreads();
    for (int c = threadIdx.x; c < kR; c += blockDim.x) {
        float acc = 0.f;
        for (int s = 0; s < nsplit; ++s) acc += w[s] * part[(((size_t) t * nsplit + s) * 64 + h) * kR + c];
        ctx[((size_t) t * 64 + h) * kR + c] = acc * inv;
    }
}

}  // namespace

size_t mla_tc_part_floats(int T, int nsplit) { return nsplit > 1 ? (size_t) T * nsplit * 64 * (kR + 2) : 0; }

void mla_attend_tc(const float* qa, const float* lat, const int32_t* sel, const int32_t* cnt, int sel_ld, int pos0,
                   float scale, float* ctx, int T, int nsplit, float* part, cudaStream_t s) {
    static bool attr = false;
    if (!attr) {
        cudaFuncSetAttribute(mla_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) kSmem);
        attr = true;
    }
    nsplit = nsplit < 1 ? 1 : (nsplit > 64 ? 64 : nsplit);
    mla_tc_kernel<<<dim3((unsigned) T, 64 / kHG, (unsigned) nsplit), kThreads, kSmem, s>>>(qa, lat, sel, cnt, sel_ld, pos0,
                                                                                          scale, ctx, part, nsplit);
    ck("mla_tc");
    if (nsplit > 1) {
        mla_merge_kernel<<<dim3((unsigned) T, 64), 128, 0, s>>>(part, ctx, nsplit);
        ck("mla_merge");
    }
}

}  // namespace glm

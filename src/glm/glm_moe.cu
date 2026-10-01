// src/glm/glm_moe.cu - GLM's pieces around Strata's MMQ expert products in the prompt path (moe_mmq.hpp).
//
// A group of n experts sits in consecutive staging slots `stride` bytes apart ([gate | up | down | tail] blobs);
// its rows (entries in expert order) run r0..r0+nr, expert j owning [r0 + rb[j], r0 + rb[j+1]). MMQ writes the
// gate/up products unscaled; each expert's weight_scale_2 values live in its blob's 16-byte tail
// {s_gate, s_up, s_down, 0}.
#include "glm_kernels.cuh"

#include <cstdio>
#include <cstdlib>

namespace glm {
namespace {

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm moe %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__device__ __forceinline__ float ue4m3_raw(int x) {   // the scale without ggml's 0.5 (it cancels here)
    if (x <= 0 || x >= 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF, m = x & 7;
    return e == 0 ? ldexpf((float) m, -9) : ldexpf(1.0f + (float) m / 8.0f, e - 7);
}

// a thread per 16 values: decode (doubled E2M1 x scale), then the scale (of 3 around amax / 3) whose {0,1,2,3}
// rounding has the least squared error; codes 0..3 are the doubled E2M1 values 0..3, +8 for the sign
__global__ void fake3_kernel(const unsigned long long* grp_ptr, const int32_t* n_groups, int cap, size_t nvb, size_t tail_off) {
    const int g = blockIdx.y;
    if (g >= min(cap, n_groups[0])) return;
    uint8_t* blob = (uint8_t*) grp_ptr[g];
    if (((const float*) (blob + tail_off))[3] != 0.0f) return;   // already done
    const size_t sb = (size_t) blockIdx.x * blockDim.x + threadIdx.x;   // a sub-block of 16
    if (sb >= nvb * 4) return;
    uint8_t* b = blob + (sb / 4) * 36;   // {d[4], qs[32]}
    const int s = (int) (sb % 4);
    const float kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const float d = ue4m3_raw(b[s]);
    float x[16], amax = 0.f;
    for (int j = 0; j < 8; ++j) {
        const uint8_t q = b[4 + s * 8 + j];
        x[j] = kv[q & 15] * d;
        x[j + 8] = kv[q >> 4] * d;
    }
    for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fabsf(x[j]));
    if (amax == 0.f) return;
    int c0 = 1;   // the UE4M3 code nearest amax / 3
    float best_d = 1e30f;
    for (int c = 1; c < 0x7F; ++c) {
        const float r = fabsf(ue4m3_raw(c) - amax / 3.f);
        if (r < best_d) { best_d = r; c0 = c; }
    }
    int bc = c0;
    float be = 1e30f;
    for (int c = max(1, c0 - 1); c <= min(0x7E, c0 + 1); ++c) {
        const float dn = ue4m3_raw(c);
        float e = 0.f;
        for (int j = 0; j < 16; ++j) {
            const float k = fminf(rintf(fabsf(x[j]) / dn), 3.f);
            const float dd = fabsf(x[j]) - k * dn;
            e += dd * dd;
        }
        if (e < be) { be = e; bc = c; }
    }
    const float dn = ue4m3_raw(bc);
    b[s] = (uint8_t) bc;
    for (int j = 0; j < 8; ++j) {
        const int k0 = (int) fminf(rintf(fabsf(x[j]) / dn), 3.f), k1 = (int) fminf(rintf(fabsf(x[j + 8]) / dn), 3.f);
        const int c0q = k0 | (x[j] < 0.f && k0 ? 8 : 0), c1q = k1 | (x[j + 8] < 0.f && k1 ? 8 : 0);
        b[4 + s * 8 + j] = (uint8_t) (c0q | (c1q << 4));
    }
}

__global__ void fake3_mark_kernel(const unsigned long long* grp_ptr, const int32_t* n_groups, int cap, size_t tail_off) {
    const int g = threadIdx.x;
    if (g >= min(cap, n_groups[0])) return;
    ((float*) ((uint8_t*) grp_ptr[g] + tail_off))[3] = 1.0f;
}

__global__ void gather_tails_kernel(const uint8_t* base, size_t stride, size_t tail_off, int n, float* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 4 * n) return;
    out[i] = ((const float*) (base + (size_t) (i / 4) * stride + tail_off))[i % 4];
}

__device__ __forceinline__ int expert_of(const int32_t* rb, int n, int rr) {   // rb relative, n + 1 entries
    int j = 0;
    while (j + 1 < n && rr >= rb[j + 1]) ++j;
    return j;
}

__global__ void swiglu_rows_kernel(const float* __restrict__ gu, float* __restrict__ h, const int32_t* __restrict__ rb,
                                   int n, const float* __restrict__ tails, long long r0, long long nrows, int ff, float lim) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x, rr = i / ff;   // rr: row within the group
    const int c = (int) (i % ff);
    if (rr >= nrows) return;
    const int j = expert_of(rb, n, (int) rr);
    const float* g = gu + (size_t) (r0 + rr) * 2 * ff;
    float gg = g[c] * tails[4 * j], uu = g[ff + c] * tails[4 * j + 1];
    if (lim > 0.f) { gg = fminf(gg, lim); uu = fminf(fmaxf(uu, -lim), lim); }
    h[(size_t) (r0 + rr) * ff + c] = gg / (1.f + expf(-gg)) * uu;
}

__global__ void scale_entry_wts_kernel(float* wts, const int32_t* dst, const int32_t* rb, int n, const float* tails, int nr) {
    const int rr = blockIdx.x * blockDim.x + threadIdx.x;
    if (rr >= nr) return;
    wts[dst[rr]] *= tails[4 * expert_of(rb, n, rr) + 2];
}

}  // namespace

void fake3_groups(const unsigned long long* grp_ptr, const int32_t* n_groups, int cap, size_t nvb, size_t tail_off,
                  cudaStream_t s) {
    if (cap <= 0) return;
    fake3_kernel<<<dim3((unsigned) ((nvb * 4 + 255) / 256), (unsigned) cap), 256, 0, s>>>(grp_ptr, n_groups, cap, nvb, tail_off);
    fake3_mark_kernel<<<1, 32, 0, s>>>(grp_ptr, n_groups, cap, tail_off);
    ck("fake3");
}

void gather_tails(const uint8_t* base, size_t stride, size_t tail_off, int n, float* out, cudaStream_t s) {
    gather_tails_kernel<<<(unsigned) ((4 * n + 63) / 64), 64, 0, s>>>(base, stride, tail_off, n, out);
    ck("gather_tails");
}

void swiglu_rows(const float* gu, float* h, const int32_t* rb, int n, const float* tails, long long r0, int nr, int ff,
                 float lim, cudaStream_t s) {
    if (nr <= 0) return;
    const long long tot = (long long) nr * ff;
    swiglu_rows_kernel<<<(unsigned) ((tot + 255) / 256), 256, 0, s>>>(gu, h, rb, n, tails, r0, nr, ff, lim);
    ck("swiglu_rows");
}

void scale_entry_wts(float* wts, const int32_t* dst, const int32_t* rb, int n, const float* tails, int nr, cudaStream_t s) {
    if (nr <= 0) return;
    scale_entry_wts_kernel<<<(unsigned) ((nr + 255) / 256), 256, 0, s>>>(wts, dst, rb, n, tails, nr);
    ck("scale_entry_wts");
}

}  // namespace glm

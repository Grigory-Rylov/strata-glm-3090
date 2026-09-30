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

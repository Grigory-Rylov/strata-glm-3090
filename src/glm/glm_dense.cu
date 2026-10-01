// src/glm/glm_dense.cu - strata-glm's dense weights in FP8 (see Mat in glm_kernels.cuh).
//
// The dense projections are the whole per-token read besides the experts (~17 GB in BF16): E4M3 with a scale per
// row halves it and leaves ~8.5 GB of VRAM to the expert tier. Quantized once at load; the prompt path turns a
// matrix back into BF16 exactly and puts the row scale on its output columns.
#include "glm_kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

#include <cstdio>
#include <cstdlib>

namespace glm {
namespace {

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "glm dense %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

__device__ __forceinline__ float bfv(bf16 v) { return __uint_as_float((uint32_t) v << 16); }

__device__ __forceinline__ float2 e4m3x2(uint32_t two) {   // two E4M3 bytes (low one first) -> floats
    const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) (two & 0xffffu), __NV_E4M3);
    return __half22float2(*(const __half2*) &h);
}

__device__ __forceinline__ float wsum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// a warp per row, 16 weights per lane per step
template <int NT>
__global__ void __launch_bounds__(256) gemv_fp8_kernel(const uint8_t* __restrict__ W, const float* __restrict__ scale,
                                                       const float* __restrict__ x, float* __restrict__ y, int rows,
                                                       int cols, int x_ld, int y_ld) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint4* wr = (const uint4*) (W + (size_t) row * cols);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int c16 = lane; c16 < cols / 16; c16 += 32) {
        const uint4 u = __ldg(wr + c16);
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
        float wf[16];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = e4m3x2(w4[j]), b = e4m3x2(w4[j] >> 16);
            wf[4 * j] = a.x; wf[4 * j + 1] = a.y; wf[4 * j + 2] = b.x; wf[4 * j + 3] = b.y;
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4* xv = (const float4*) (x + (size_t) t * x_ld + (size_t) c16 * 16);
            float a = 0.f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 v = xv[q];
                a += wf[4 * q] * v.x + wf[4 * q + 1] * v.y + wf[4 * q + 2] * v.z + wf[4 * q + 3] * v.w;
            }
            acc[t] += a;
        }
    }
    const float sc = scale[row];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float v = wsum(acc[t]);
        if (lane == 0) y[(size_t) t * y_ld + row] = v * sc;
    }
}

__device__ __forceinline__ float2 e2m1x2(uint32_t byte) {   // two e2m1 codes (the low nibble first) -> floats
    const __half2_raw h = __nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t) (byte & 0xffu), __NV_E2M1);
    return __half22float2(*(const __half2*) &h);
}

// a warp per row, 32 weights (16 bytes, two blocks) per lane per step; each block's dot product times its scale
template <int NT>
__global__ void __launch_bounds__(256) gemv_nvfp4_kernel(const uint8_t* __restrict__ W, const uint8_t* __restrict__ bsc,
                                                         const float* __restrict__ scale, const float* __restrict__ x,
                                                         float* __restrict__ y, int rows, int cols, int x_ld, int y_ld) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint4* wr = (const uint4*) (W + (size_t) row * (cols / 2));
    const uint16_t* sr = (const uint16_t*) (bsc + (size_t) row * (cols / 16));
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int c32 = lane; c32 < cols / 32; c32 += 32) {
        const uint4 u = __ldg(wr + c32);
        const float2 bs = e4m3x2(__ldg(sr + c32));
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
        float wf[32];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float2 v = e2m1x2(w4[j] >> (8 * b));
                wf[8 * j + 2 * b] = v.x;
                wf[8 * j + 2 * b + 1] = v.y;
            }
        }
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4* xv = (const float4*) (x + (size_t) t * x_ld + (size_t) c32 * 32);
            float a0 = 0.f, a1 = 0.f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 v = xv[q], v2 = xv[q + 4];
                a0 += wf[4 * q] * v.x + wf[4 * q + 1] * v.y + wf[4 * q + 2] * v.z + wf[4 * q + 3] * v.w;
                a1 += wf[16 + 4 * q] * v2.x + wf[16 + 4 * q + 1] * v2.y + wf[16 + 4 * q + 2] * v2.z + wf[16 + 4 * q + 3] * v2.w;
            }
            acc[t] += a0 * bs.x + a1 * bs.y;
        }
    }
    const float sc = scale[row];
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float v = wsum(acc[t]);
        if (lane == 0) y[(size_t) t * y_ld + row] = v * sc;
    }
}

// a warp per row, 16 weights per lane per step (a 32-value block's scale shared by two lanes)
template <int NT>
__global__ void __launch_bounds__(256) gemv_i8_kernel(const int8_t* __restrict__ W, const __half* __restrict__ qs,
                                                      const float* __restrict__ x, float* __restrict__ y, int rows,
                                                      int cols, int x_ld, int y_ld) {
    const int row = blockIdx.x * 8 + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= rows) return;
    const uint4* wr = (const uint4*) (W + (size_t) row * cols);
    const __half* sr = qs + (size_t) row * (cols / 32);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; ++t) acc[t] = 0.f;
    for (int c16 = lane; c16 < cols / 16; c16 += 32) {
        const uint4 u = __ldg(wr + c16);
        const float sc = __half2float(sr[c16 >> 1]);
        const uint32_t w4[4] = {u.x, u.y, u.z, u.w};
        float wf[16];
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
            for (int b = 0; b < 4; ++b) wf[4 * j + b] = (float) (int8_t) (w4[j] >> (8 * b));
#pragma unroll
        for (int t = 0; t < NT; ++t) {
            const float4* xv = (const float4*) (x + (size_t) t * x_ld + (size_t) c16 * 16);
            float a = 0.f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float4 v = xv[q];
                a += wf[4 * q] * v.x + wf[4 * q + 1] * v.y + wf[4 * q + 2] * v.z + wf[4 * q + 3] * v.w;
            }
            acc[t] += a * sc;
        }
    }
#pragma unroll
    for (int t = 0; t < NT; ++t) {
        const float v = wsum(acc[t]);
        if (lane == 0) y[(size_t) t * y_ld + row] = v;
    }
}

__global__ void quant_i8_kernel(const bf16* __restrict__ w, size_t nblk, int8_t* __restrict__ q, __half* __restrict__ qs) {
    const size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x;   // one 32-value block (rows are whole blocks)
    if (b >= nblk) return;
    float v[32], m = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) {
        v[i] = bfv(w[b * 32 + i]);
        m = fmaxf(m, fabsf(v[i]));
    }
    const __half hs = __float2half(m / 127.f);
    qs[b] = hs;
    const float sc = __half2float(hs), inv = sc > 0.f ? 1.f / sc : 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) q[b * 32 + i] = (int8_t) __float2int_rn(fminf(fmaxf(v[i] * inv, -127.f), 127.f));
}

__global__ void i8_to_bf16_kernel(const int8_t* __restrict__ q, const __half* __restrict__ qs, bf16* __restrict__ out, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t u = __float_as_uint((float) q[i] * __half2float(qs[i / 32]));
    u += 0x7fffu + ((u >> 16) & 1u);   // round to nearest even
    out[i] = (bf16) (u >> 16);
}

__device__ __forceinline__ int e2m1_code(float v) {   // the nearest e2m1 value's code (sign in bit 3)
    const float a = fabsf(v);
    const int m = a < 0.25f ? 0 : a < 0.75f ? 1 : a < 1.25f ? 2 : a < 1.75f ? 3 : a < 2.5f ? 4 : a < 3.5f ? 5 : a < 5.f ? 6 : 7;
    return m | (v < 0.f && m ? 8 : 0);
}

__device__ __forceinline__ float e2m1_val(int c) {
    const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (c & 8) ? -mag[c & 7] : mag[c & 7];
}

__device__ __forceinline__ float e4m3_val(int c) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) c, __NV_E4M3);
    return __half2float(*(const __half*) &h);
}

// a block per row: its amax gives the row scale; then a thread per 16-value block tries 4 E4M3 scales around
// amax / 6 (a lower one clips the largest value but can round the rest closer) and keeps the least squared error
__global__ void quant_nvfp4_kernel(const bf16* __restrict__ w, int cols, uint8_t* __restrict__ q, uint8_t* __restrict__ bsc,
                                   float* __restrict__ scale) {
    const size_t r = blockIdx.x;
    const bf16* wr = w + r * cols;
    float m = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, fabsf(bfv(wr[c])));
    __shared__ float part[32];
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) part[0] = v;
    }
    __syncthreads();
    const float s2 = part[0] > 0.f ? part[0] / (6.f * 448.f) : 1.f;
    if (threadIdx.x == 0) scale[r] = s2;
    for (int b = threadIdx.x; b < cols / 16; b += blockDim.x) {
        float v[16], bm = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            v[i] = bfv(wr[b * 16 + i]);
            bm = fmaxf(bm, fabsf(v[i]));
        }
        const int c0 = (int) __nv_cvt_float_to_fp8(bm / 6.f / s2, __NV_SATFINITE, __NV_E4M3);
        int best = 0;
        float best_err = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) best_err += v[i] * v[i];   // scale 0: every code 0
        for (int c = max(1, c0 - 2); c <= min(0x7e, c0 + 1); ++c) {
            const float sb = e4m3_val(c) * s2, inv = 1.f / sb;
            float err = 0.f;
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float d = v[i] - e2m1_val(e2m1_code(v[i] * inv)) * sb;
                err += d * d;
            }
            if (err < best_err) { best_err = err; best = c; }
        }
        bsc[r * (cols / 16) + b] = (uint8_t) best;
        const float inv = best ? 1.f / (e4m3_val(best) * s2) : 0.f;
#pragma unroll
        for (int i = 0; i < 16; i += 2)
            q[r * (cols / 2) + b * 8 + i / 2] = (uint8_t) (e2m1_code(v[i] * inv) | (e2m1_code(v[i + 1] * inv) << 4));
    }
}

__global__ void quant_fp8_kernel(const bf16* __restrict__ w, int cols, uint8_t* __restrict__ q, float* __restrict__ scale) {
    const size_t r = blockIdx.x;
    const bf16* wr = w + r * cols;
    float m = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) m = fmaxf(m, fabsf(bfv(wr[c])));
    __shared__ float part[32];
    m = fmaxf(m, 0.f);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = m;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < (blockDim.x >> 5) ? part[threadIdx.x] : 0.f;
        for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
        if (threadIdx.x == 0) part[0] = v;
    }
    __syncthreads();
    const float s = part[0] > 0.f ? part[0] / 448.f : 1.f;
    if (threadIdx.x == 0) scale[r] = s;
    const float inv = 1.f / s;
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        q[r * cols + c] = (uint8_t) __nv_cvt_float_to_fp8(bfv(wr[c]) * inv, __NV_SATFINITE, __NV_E4M3);
}

__global__ void fp8_to_bf16_kernel(const uint8_t* __restrict__ q, bf16* __restrict__ out, size_t n2) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;   // two values
    if (i >= n2) return;
    const float2 f = e4m3x2(((const uint16_t*) q)[i]);
    out[2 * i] = (bf16) (__float_as_uint(f.x) >> 16);       // exact: at most 4 significant bits
    out[2 * i + 1] = (bf16) (__float_as_uint(f.y) >> 16);
}

__global__ void scale_cols_kernel(float* __restrict__ Y, const float* __restrict__ scale, int N, int T, int ldy) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) N * T) return;
    const size_t t = i / N, c = i % N;
    Y[t * ldy + c] *= scale[c];
}

__device__ __forceinline__ float e4m3v(uint8_t b) {
    const __half_raw h = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) b, __NV_E4M3);
    return __half2float(*(const __half*) &h);
}

__global__ void nvfp4_to_bf16_kernel(const uint8_t* __restrict__ w, const uint8_t* __restrict__ sc, bf16* __restrict__ out,
                                     int rows, int cols) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;   // one byte = two values
    if (i >= (long long) rows * cols / 2) return;
    const float E2M1[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -0.5f, -1, -1.5f, -2, -3, -4, -6};
    const long long r = i / (cols / 2), c = 2 * (i % (cols / 2));
    const float s = e4m3v(sc[r * (cols / 16) + c / 16]);
    const uint8_t b = w[i];
    out[r * cols + c] = (bf16) (__float_as_uint(E2M1[b & 0xF] * s) >> 16);   // exact: <= 5 significant bits
    out[r * cols + c + 1] = (bf16) (__float_as_uint(E2M1[b >> 4] * s) >> 16);
}

__global__ void fill_kernel(float* p, float v, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = v;
}

}  // namespace

void gemv(const Mat& W, const float* x, float* y, int rows, int cols, int nt, cudaStream_t s, int x_ld, int y_ld) {
    if (W.qs) {
        if (cols % 32) { std::fprintf(stderr, "gemv i8: cols %d not a multiple of 32\n", cols); std::exit(1); }
        x_ld = x_ld ? x_ld : cols;
        y_ld = y_ld ? y_ld : rows;
        const dim3 g((unsigned) ((rows + 7) / 8));
        const int8_t* w = (const int8_t*) W.w;
        const __half* qs = (const __half*) W.qs;
        switch (nt) {
            case 1: gemv_i8_kernel<1><<<g, 256, 0, s>>>(w, qs, x, y, rows, cols, x_ld, y_ld); break;
            case 2: gemv_i8_kernel<2><<<g, 256, 0, s>>>(w, qs, x, y, rows, cols, x_ld, y_ld); break;
            case 4: gemv_i8_kernel<4><<<g, 256, 0, s>>>(w, qs, x, y, rows, cols, x_ld, y_ld); break;
            default:
                for (int t = 0; t < nt; ++t)
                    gemv_i8_kernel<1><<<g, 256, 0, s>>>(w, qs, x + (size_t) t * x_ld, y + (size_t) t * y_ld, rows, cols, x_ld, y_ld);
        }
        ck("gemv i8");
        return;
    }
    if (W.bsc) {
        if (cols % 32) { std::fprintf(stderr, "gemv nvfp4: cols %d not a multiple of 32\n", cols); std::exit(1); }
        x_ld = x_ld ? x_ld : cols;
        y_ld = y_ld ? y_ld : rows;
        const dim3 g((unsigned) ((rows + 7) / 8));
        const uint8_t* w = (const uint8_t*) W.w;
        switch (nt) {
            case 1: gemv_nvfp4_kernel<1><<<g, 256, 0, s>>>(w, W.bsc, W.scale, x, y, rows, cols, x_ld, y_ld); break;
            case 2: gemv_nvfp4_kernel<2><<<g, 256, 0, s>>>(w, W.bsc, W.scale, x, y, rows, cols, x_ld, y_ld); break;
            case 4: gemv_nvfp4_kernel<4><<<g, 256, 0, s>>>(w, W.bsc, W.scale, x, y, rows, cols, x_ld, y_ld); break;
            default:
                for (int t = 0; t < nt; ++t)
                    gemv_nvfp4_kernel<1><<<g, 256, 0, s>>>(w, W.bsc, W.scale, x + (size_t) t * x_ld, y + (size_t) t * y_ld,
                                                          rows, cols, x_ld, y_ld);
        }
        ck("gemv nvfp4");
        return;
    }
    if (!W.fp8) {
        gemv_bf16(W.b16(), x, y, rows, cols, nt, s, x_ld, y_ld);
        if (W.scale) scale_cols(y, W.scale, rows, nt, y_ld ? y_ld : rows, s);
        return;
    }
    if (cols % 16) { std::fprintf(stderr, "gemv fp8: cols %d not a multiple of 16\n", cols); std::exit(1); }
    x_ld = x_ld ? x_ld : cols;
    y_ld = y_ld ? y_ld : rows;
    const dim3 g((unsigned) ((rows + 7) / 8));
    const uint8_t* w = (const uint8_t*) W.w;
    switch (nt) {
        case 1: gemv_fp8_kernel<1><<<g, 256, 0, s>>>(w, W.scale, x, y, rows, cols, x_ld, y_ld); break;
        case 2: gemv_fp8_kernel<2><<<g, 256, 0, s>>>(w, W.scale, x, y, rows, cols, x_ld, y_ld); break;
        case 4: gemv_fp8_kernel<4><<<g, 256, 0, s>>>(w, W.scale, x, y, rows, cols, x_ld, y_ld); break;
        case 8: gemv_fp8_kernel<8><<<g, 256, 0, s>>>(w, W.scale, x, y, rows, cols, x_ld, y_ld); break;
        default:
            for (int t = 0; t < nt; ++t)
                gemv_fp8_kernel<1><<<g, 256, 0, s>>>(w, W.scale, x + (size_t) t * x_ld, y + (size_t) t * y_ld, rows, cols, x_ld, y_ld);
    }
    ck("gemv fp8");
}

void quant_fp8_rows(const bf16* w, int rows, int cols, uint8_t* q, float* scale, cudaStream_t s) {
    quant_fp8_kernel<<<(unsigned) rows, 256, 0, s>>>(w, cols, q, scale);
    ck("quant_fp8_rows");
}

void quant_nvfp4_rows(const bf16* w, int rows, int cols, uint8_t* q, uint8_t* bsc, float* scale, cudaStream_t s) {
    quant_nvfp4_kernel<<<(unsigned) rows, 256, 0, s>>>(w, cols, q, bsc, scale);
    ck("quant_nvfp4_rows");
}

void quant_i8_rows(const bf16* w, int rows, int cols, int8_t* q, f16* qs, cudaStream_t s) {
    const size_t nblk = (size_t) rows * cols / 32;
    quant_i8_kernel<<<(unsigned) ((nblk + 127) / 128), 128, 0, s>>>(w, nblk, q, (__half*) qs);
    ck("quant_i8_rows");
}

void i8_to_bf16(const int8_t* q, const f16* qs, bf16* out, int rows, int cols, cudaStream_t s) {
    const size_t n = (size_t) rows * cols;
    i8_to_bf16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(q, (const __half*) qs, out, n);
    ck("i8_to_bf16");
}

void fp8_to_bf16(const uint8_t* q, bf16* out, size_t n, cudaStream_t s) {
    const size_t n2 = n / 2;
    fp8_to_bf16_kernel<<<(unsigned) ((n2 + 255) / 256), 256, 0, s>>>(q, out, n2);
    ck("fp8_to_bf16");
}

void scale_cols(float* Y, const float* scale, int N, int T, int ldy, cudaStream_t s) {
    const size_t n = (size_t) N * T;
    scale_cols_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(Y, scale, N, T, ldy);
    ck("scale_cols");
}

void nvfp4_to_bf16(const uint8_t* w, const uint8_t* sc, bf16* out, int rows, int cols, cudaStream_t s) {
    const long long n = (long long) rows * cols / 2;
    nvfp4_to_bf16_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(w, sc, out, rows, cols);
    ck("nvfp4_to_bf16");
}

void fill(float* p, float v, int n, cudaStream_t s) {
    fill_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s>>>(p, v, n);
    ck("fill");
}

}  // namespace glm

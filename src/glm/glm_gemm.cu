// src/glm/glm_gemm.cu - see glm_gemm.cuh.
#include "glm_gemm.cuh"

#include <cstdio>
#include <cstdlib>

namespace glm {
namespace {

void ckb(cublasStatus_t st, const char* what) {
    if (st != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "glm gemm %s: cuBLAS status %d\n", what, (int) st);
        std::exit(1);
    }
}

__device__ __forceinline__ uint16_t to_bf16_rne(float f) {
    uint32_t u = __float_as_uint(f);
    u += 0x7fffu + ((u >> 16) & 1u);   // round to nearest even (finite inputs)
    return (uint16_t) (u >> 16);
}

__global__ void split_kernel(const float* __restrict__ x, int rows, int cols, int ldx, uint16_t* __restrict__ hi,
                             uint16_t* __restrict__ lo) {
    const long long i = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long) rows * cols) return;
    const long long r = i / cols, c = i % cols;
    const float v = x[r * ldx + c];
    const uint16_t h = to_bf16_rne(v);
    hi[i] = h;
    lo[i] = to_bf16_rne(v - __uint_as_float((uint32_t) h << 16));
}

}  // namespace

void Gemm::init(cudaStream_t s, size_t max_elems) {
    s_ = s;
    if (!h_) ckb(cublasCreate(&h_), "create");   // release() keeps the handle; a re-init reuses it
    ckb(cublasSetStream(h_, s), "stream");
    ckb(cublasSetMathMode(h_, CUBLAS_DEFAULT_MATH), "math mode");   // no TF32 for the FP32 path
    cap_ = max_elems;
    if (cudaMalloc(&hi_, cap_ * sizeof(bf16)) != cudaSuccess || cudaMalloc(&lo_, cap_ * sizeof(bf16)) != cudaSuccess) {
        std::fprintf(stderr, "glm gemm: cannot allocate %zu activation elements\n", cap_);
        std::exit(1);
    }
}

void Gemm::split(const float* X, int rows, int cols, int ldx) {
    const long long n = (long long) rows * cols;
    if ((size_t) n > cap_) { std::fprintf(stderr, "glm gemm: %lld activation elements > %zu\n", n, cap_); std::exit(1); }
    split_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, s_>>>(X, rows, cols, ldx ? ldx : cols, hi_, lo_);
}

void Gemm::w16(const bf16* W, const float* X, float* Y, int N, int K, int T, int ldx, int ldy) {
    split(X, T, K, ldx);
    const float one = 1.f, zero = 0.f;
    // row-major Y[T][N] = X[T][K] W^T  <=>  column-major Y^T (N x T) = W^T... : W is K x N column-major (ld K), op T
    ckb(cublasGemmEx(h_, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, W, CUDA_R_16BF, K, hi_, CUDA_R_16BF, K, &zero, Y,
                     CUDA_R_32F, ldy ? ldy : N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "bf16 hi");
    ckb(cublasGemmEx(h_, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, W, CUDA_R_16BF, K, lo_, CUDA_R_16BF, K, &one, Y,
                     CUDA_R_32F, ldy ? ldy : N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "bf16 lo");
}

void Gemm::wmat(const Mat& W, const float* X, float* Y, int N, int K, int T, int ldx, int ldy) {
    const bf16* w = W.b16();
    if (W.fp8 || W.bsc || W.qs) {
        const size_t n = (size_t) N * K;
        if (n > wcap_) {
            if (wbuf_) cudaFree(wbuf_);
            if (cudaMalloc(&wbuf_, n * sizeof(bf16)) != cudaSuccess) {
                std::fprintf(stderr, "glm gemm: cannot allocate %zu BF16 weights\n", n);
                std::exit(1);
            }
            wcap_ = n;
        }
        if (W.qs) i8_to_bf16((const int8_t*) W.w, W.qs, wbuf_, N, K, s_);
        else if (W.bsc) nvfp4_to_bf16((const uint8_t*) W.w, W.bsc, wbuf_, N, K, s_);
        else fp8_to_bf16((const uint8_t*) W.w, wbuf_, n, s_);
        w = wbuf_;
    }
    w16(w, X, Y, N, K, T, ldx, ldy);
    if (W.scale) scale_cols(Y, W.scale, N, T, ldy ? ldy : N, s_);
}

void Gemm::w32(const float* W, const float* X, float* Y, int N, int K, int T) {
    const float one = 1.f, zero = 0.f;
    ckb(cublasSgemm(h_, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, W, K, X, K, &zero, Y, N), "f32");
}

void Gemm::heads16(const bf16* W, long long w_stride, bool transW, const float* X, int ldx, float* Y, int ldy,
                      int Nd, int Kd, int T, int H) {
    // X: H blocks of Kd per row (row stride ldx); split packs them as [T][H*Kd] with the same interleaving
    split(X, T, H * Kd, ldx);
    const float one = 1.f, zero = 0.f;
    // column-major: Y_h^T (Nd x T) = op(W_h) (Nd x Kd) * X_h^T (Kd x T)
    //   !transW: W_h row-major [Kd][Nd] = column-major Nd x Kd (ld Nd), op N
    //    transW: W_h row-major [Nd][Kd] = column-major Kd x Nd (ld Kd), op T
    const cublasOperation_t opw = transW ? CUBLAS_OP_T : CUBLAS_OP_N;
    const int ldw = transW ? Kd : Nd;
    for (int pass = 0; pass < 2; ++pass)
        ckb(cublasGemmStridedBatchedEx(h_, opw, CUBLAS_OP_N, Nd, T, Kd, &one, W, CUDA_R_16BF, ldw, w_stride,
                                       pass ? lo_ : hi_, CUDA_R_16BF, H * Kd, Kd, pass ? &one : &zero, Y, CUDA_R_32F,
                                       ldy, Nd, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT), "bf16 heads");
}

}  // namespace glm

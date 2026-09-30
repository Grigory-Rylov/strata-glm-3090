// src/glm/glm_gemm.cuh - batched projections for strata-glm's prompt path (cuBLAS).
//
// BF16 weights times FP32 activations: x is split into hi = bf16(x) and lo = bf16(x - hi) and both are multiplied
// in FP32 (two GEMMs, the second accumulating), so x keeps ~16 mantissa bits - what decode's FP32 GEMV sees to
// within ~1e-5. FP32 weights (the leading dense MLP) go through SGEMM without TF32.
#pragma once

#include "glm_kernels.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

namespace glm {

class Gemm {
public:
    void init(cudaStream_t s, size_t max_elems);   // max_elems: the largest activation (T * K) split into hi/lo
    /// Y[t][n] = sum_k W[n][k] * X[t][k]; W BF16 [N][K], X FP32 [T][K] (row stride ldx, 0 = K), Y FP32 [T][N]
    /// (row stride ldy, 0 = N).
    void w16(const bf16* W, const float* X, float* Y, int N, int K, int T, int ldx = 0, int ldy = 0);
    /// the same with FP32 weights
    void w32(const float* W, const float* X, float* Y, int N, int K, int T);
    /// the same with a Mat: an FP8 one turned into BF16 (exactly) in a scratch buffer, the row scale on Y's columns
    void wmat(const Mat& W, const float* X, float* Y, int N, int K, int T, int ldx = 0, int ldy = 0);
    /// per head h < H: Y_h[t][:] = X_h[t][:] (Kd) times W_h (row-major [Kd][Nd] when !transW, [Nd][Kd] when transW),
    /// W_h = W + h * w_stride; X_h = X + h * Kd (rows ldx apart); Y_h = Y + h * Nd (rows ldy apart)
    void heads16(const bf16* W, long long w_stride, bool transW, const float* X, int ldx, float* Y, int ldy, int Nd,
                    int Kd, int T, int H);
    cublasHandle_t handle() const { return h_; }
    void release() {   // the activation splits and the FP8 scratch (the prompt is done)
        if (hi_) cudaFree(hi_);
        if (lo_) cudaFree(lo_);
        if (wbuf_) cudaFree(wbuf_);
        hi_ = lo_ = wbuf_ = nullptr;
        cap_ = wcap_ = 0;
    }

private:
    void split(const float* X, int rows, int cols, int ldx);   // X -> hi_, lo_ (packed rows of cols)
    cublasHandle_t h_ = nullptr;
    cudaStream_t s_ = nullptr;
    bf16 *hi_ = nullptr, *lo_ = nullptr;
    size_t cap_ = 0;
    bf16* wbuf_ = nullptr;   // an FP8 matrix as BF16
    size_t wcap_ = 0;
};

}  // namespace glm

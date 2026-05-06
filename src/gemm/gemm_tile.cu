#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8


__global__ void gemm_kernel(const float *A, const float *B, float *C,
                            int M, int N, int K)
{
    __shared__ float tileA[BM][BK];
    __shared__ float tileB[BK][BN];

    int tid         = threadIdx.y * blockDim.x + threadIdx.x;  
    int num_threads = blockDim.x * blockDim.y;         

    int row_base = blockIdx.y * BM;
    int col_base = blockIdx.x * BN;

    float result[TM][TN] = {};

    for (int k = 0; k < (K + BK - 1) / BK; ++k)
    {
        // Load tileA
        for (int i = 0; i < BM * BK / num_threads; ++i) {
            int idx   = i * num_threads + tid;
            int a_row = idx / BK;
            int a_col = idx % BK;
            tileA[a_row][a_col] = (row_base + a_row < M && k * BK + a_col < K)
                ? A[(row_base + a_row) * K + k * BK + a_col] : 0.0f;
        }
        // Load tileB
        for (int i = 0; i < BK * BN / num_threads; ++i) {
            int idx   = i * num_threads + tid;
            int b_row = idx / BN;
            int b_col = idx % BN;
            tileB[b_row][b_col] = (k * BK + b_row < K && col_base + b_col < N)
                ? B[(k * BK + b_row) * N + col_base + b_col] : 0.0f;
        }
        __syncthreads();
        // compute
        for (int kk = 0; kk < BK; ++kk) {
            float regA[TM], regB[TN];
            for (int j = 0; j < TM; ++j)
                regA[j] = tileA[threadIdx.y * TM + j][kk];
            for (int i = 0; i < TN; ++i)
                regB[i] = tileB[kk][threadIdx.x * TN + i];
            for (int j = 0; j < TM; ++j)
                for (int i = 0; i < TN; ++i)
                    result[j][i] += regA[j] * regB[i];
        }
        __syncthreads();
    }

    for (int j = 0; j < TM; ++j) {
        for (int i = 0; i < TN; ++i) {
            int C_row = row_base + threadIdx.y * TM + j;
            int C_col = col_base + threadIdx.x * TN + i;
            if (C_row < M && C_col < N)
                C[C_row * N + C_col] = result[j][i];
        }
    }
}

void gemm(const float *A, const float *B, float *C,
          int M, int N, int K, cudaStream_t stream)
{
    dim3 block(BN/TN, BM/TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

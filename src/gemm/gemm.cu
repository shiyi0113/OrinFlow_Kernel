#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 16
#define TM 8
#define TN 8

#define NUM_THREADS ((BM / TM) * (BN / TN))  // 256

#define LOADS_A (BM * BK / NUM_THREADS)  // 8
#define LOADS_B (BK * BN / NUM_THREADS)  // 8

__global__ __launch_bounds__(256, 2)
void gemm_kernel(const float* A, const float* B, float* C,
                            int M, int N, int K)
{

    __shared__ float tileA[2][BM][BK + 2];
    __shared__ float tileB[2][BK][BN];

    int tid      = threadIdx.y * blockDim.x + threadIdx.x;
    int row_base = blockIdx.y * BM;
    int col_base = blockIdx.x * BN;
    int num_tiles = (K + BK - 1) / BK;

    float result[TM][TN] = {};

    #pragma unroll
    for (int i = 0; i < LOADS_A; ++i) {
        int idx = i * NUM_THREADS + tid;
        int r = idx / BK, c = idx % BK;
        tileA[0][r][c] = (row_base + r < M && c < K) ? A[(row_base + r) * K + c] : 0.f;
    }
    #pragma unroll
    for (int i = 0; i < LOADS_B; ++i) {
        int idx = i * NUM_THREADS + tid;
        int r = idx / BN, c = idx % BN;
        tileB[0][r][c] = (r < K && col_base + c < N) ? B[r * N + col_base + c] : 0.f;
    }
    __syncthreads();

    for (int k = 0; k < num_tiles; ++k) {
        int cur = k & 1;
        int nxt = cur ^ 1;

        if (k + 1 < num_tiles) {
            int k1 = k + 1;
            #pragma unroll
            for (int i = 0; i < LOADS_A; ++i) {
                int idx = i * NUM_THREADS + tid;
                int r = idx / BK, c = idx % BK;
                tileA[nxt][r][c] = (row_base + r < M && k1 * BK + c < K)
                    ? A[(row_base + r) * K + k1 * BK + c] : 0.f;
            }
            #pragma unroll
            for (int i = 0; i < LOADS_B; ++i) {
                int idx = i * NUM_THREADS + tid;
                int r = idx / BN, c = idx % BN;
                tileB[nxt][r][c] = (k1 * BK + r < K && col_base + c < N)
                    ? B[(k1 * BK + r) * N + col_base + c] : 0.f;
            }
        }

        #pragma unroll
        for (int kk = 0; kk < BK; ++kk) {
            float regA[TM], regB[TN];
            #pragma unroll
            for (int j = 0; j < TM; ++j)
                regA[j] = tileA[cur][threadIdx.y * TM + j][kk];
            #pragma unroll
            for (int i = 0; i < TN; ++i)
                regB[i] = tileB[cur][kk][threadIdx.x * TN + i];
            #pragma unroll
            for (int j = 0; j < TM; ++j)
                #pragma unroll
                for (int i = 0; i < TN; ++i)
                    result[j][i] += regA[j] * regB[i];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int j = 0; j < TM; ++j) {
        #pragma unroll
        for (int i = 0; i < TN; ++i) {
            int C_row = row_base + threadIdx.y * TM + j;
            int C_col = col_base + threadIdx.x * TN + i;
            if (C_row < M && C_col < N)
                C[C_row * N + C_col] = result[j][i];
        }
    }
}

void gemm(const float* A, const float* B, float* C,
          int M, int N, int K, cudaStream_t stream)
{
    dim3 block(BN / TN, BM / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}

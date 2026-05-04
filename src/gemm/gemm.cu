#include <cuda_runtime.h>

__global__
void gemm_kernel(const float* A, const float* B, float* C,
                 int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float sum = 0.0f;
        for(int i = 0; i < K; ++i){
            float a_val = A[row * K + i];
            float b_val = B[i * N + col];
            sum += a_val * b_val;
        }
        C[row * N + col] = sum;
    }
}

void gemm(const float* A, const float* B, float* C,
          int M, int N, int K, cudaStream_t stream)
{
    dim3 block(32,16);
    dim3  grid((N + block.x - 1)/block.x , (M + block.y - 1)/block.y);
    gemm_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
}
#include <cuda_runtime.h>

static __device__ 
float warp_reduce_shfl(float val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__global__ 
void reduce_kernel(const float *x, float *y, int n)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
    float a = (idx < n)                  ? x[idx]                : 0.0f;
    float b = (idx + blockDim.x < n)     ? x[idx + blockDim.x]   : 0.0f;
    float c = (idx + blockDim.x * 2 < n) ? x[idx + blockDim.x*2] : 0.0f;
    float d = (idx + blockDim.x * 3 < n) ? x[idx + blockDim.x*3] : 0.0f;
    float e = (idx + blockDim.x * 4 < n) ? x[idx + blockDim.x*4] : 0.0f;
    float f = (idx + blockDim.x * 5 < n) ? x[idx + blockDim.x*5] : 0.0f;
    float g = (idx + blockDim.x * 6 < n) ? x[idx + blockDim.x*6] : 0.0f;
    float h = (idx + blockDim.x * 7 < n) ? x[idx + blockDim.x*7] : 0.0f;
    
    float sum = (a+b) + (c+d) + (e+f) + (g+h);
    int lane   = tid % 32;
    int warpId = tid / 32;
    
    sum = warp_reduce_shfl(sum);
    if(lane == 0) smem[warpId] = sum;
    __syncthreads();

    int numWarps = blockDim.x / 32;
    if(warpId == 0){
        sum = (lane < numWarps) ? smem[lane] : 0.0f;
        sum = warp_reduce_shfl(sum);
    }

    if(tid == 0) atomicAdd(y, sum);
}
void reduce(const float *d_x, float *d_y, int n, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (n + block_size*8 - 1) / (block_size*8);
    reduce_kernel<<<grid_size, block_size, block_size/32 * sizeof(float), stream>>>(d_x, d_y, n);
}
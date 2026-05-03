#include <cuda_runtime.h>

static __device__
float warp_reduce(float val){
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

__global__ void reduce_kernel(const float4 *x, float *out, int n)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x*2 + threadIdx.x;
    float4 a = (idx < n) ? x[idx] : make_float4(0.0,0.0,0.0,0.0);
    float4 b = (idx+blockDim.x < n)? x[idx + blockDim.x] : make_float4(0.0,0.0,0.0,0.0);

    smem[tid] = (a.x + b.x) + (a.y + b.y) + (a.z + b.z) + (a.w + b.w);
    __syncthreads();

    for (int s = blockDim.x/2; s > 32; s >>= 1)
    {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }

    if(tid < 32){
        float val = smem[tid] + smem[tid + 32];
        val = warp_reduce(val);
        if(tid == 0) atomicAdd(out, val);
    }
}
void reduce(const float *d_x, float *d_y, int n, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = (n + block_size * 8 - 1) / (block_size * 8);
    reduce_kernel<<<grid_size, block_size, block_size * sizeof(float), stream>>>(
        reinterpret_cast<const float4*>(d_x), d_y, n);
}
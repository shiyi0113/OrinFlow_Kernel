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
void reduce_kernel(const float* x, float* y, int n)
{
    extern __shared__ float smem[];
    
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warpId = tid >> 5;

    const float4* x4 = reinterpret_cast<const float4*>(x);
    int n4 = n >> 2; 

    int idx = blockIdx.x * (blockDim.x * 2) + tid;

    float4 va = (idx < n4)              ? x4[idx]              : make_float4(0,0,0,0);
    float4 vb = (idx + blockDim.x < n4) ? x4[idx + blockDim.x] : make_float4(0,0,0,0);

    float sum = (va.x + va.y) + (va.z + va.w)
              + (vb.x + vb.y) + (vb.z + vb.w);

    // tail: at most 3 elements, handled by first threads of block 0
    int tail = n & 3;
    if (blockIdx.x == 0 && tid < tail)
        sum += x[(n4 << 2) + tid];

    sum = warp_reduce_shfl(sum);
    if (lane == 0) smem[warpId] = sum;
    __syncthreads();

    int numWarps = blockDim.x / 32;
    if (warpId == 0) {
        sum = (lane < numWarps) ? smem[lane] : 0.0f;
        sum = warp_reduce_shfl(sum);
    }
    if (tid == 0) atomicAdd(y, sum);
}

void reduce(const float* d_x, float* d_y, int num, cudaStream_t stream){
    int block_size = 256;
    int grid_size = (num + block_size*8 - 1)/(block_size*8);
    reduce_kernel<<<grid_size, block_size, block_size/32 * sizeof(float)>>>(d_x, d_y, num);
}

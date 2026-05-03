#include <cuda_runtime.h>
#include <cfloat>

struct MD{
    float M;
    float D;
};

__device__
MD md_combine(MD a, MD b){
    MD r;
    r.M = max(a.M, b.M);
    r.D = a.D * expf(a.M - r.M) + b.D * expf(b.M - r.M);
    return r;
}

__global__
void softmax_kernel(const float* x, float* out, int m, int n)
{
    const float* x_row = x + blockIdx.x * n;
    float* out_row     = out + blockIdx.x * n;
    extern __shared__ MD smem[];

    int tid = threadIdx.x;
    MD local = {-FLT_MAX, 0.0f};
    for(int i = tid; i < n; i += blockDim.x){
        local = md_combine(local, {x_row[i], 1.0f});
    }
    smem[tid] = local;
    __syncthreads();
    for(int s = blockDim.x/2; s > 0; s >>= 1)
    {
        if(tid < s)
            smem[tid] = md_combine(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    for(int i = tid; i < n; i += blockDim.x)
        out_row[i] = expf(x_row[i] - smem[0].M) / smem[0].D;
}

void softmax(const float* x, float* out, int m, int n, cudaStream_t stream)
{
    int block_size = 256;
    int grid_size = m;
    softmax_kernel<<<grid_size, block_size, block_size*sizeof(MD), stream>>>(x, out, m, n);
}

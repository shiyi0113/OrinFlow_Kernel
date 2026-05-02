#include <cuda_runtime.h>

__global__ 
void add_kernel(const float4* a, const float4* b, float4* out, size_t size) {
    size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if(idx >= size) return;
    float4 r_a = a[idx];
    float4 r_b = b[idx];
    out[idx] = make_float4(r_a.x + r_b.x, r_a.y + r_b.y, r_a.z + r_b.z, r_a.w + r_b.w);
}

void launch_add_kernel_float(const float* a, const float* b, 
                             float* out, size_t size, 
                             cudaStream_t stream) 
{
    int block_size = 256;
    int grid_size = (size/4 + block_size - 1) / block_size;
    add_kernel<<<grid_size, block_size, 0, stream>>>(
        reinterpret_cast<const float4*>(a), 
        reinterpret_cast<const float4*>(b), 
        reinterpret_cast<float4*>(out), size/4);
}
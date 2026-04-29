#include <cuda_runtime.h>
#include <stddef.h>


__global__ void add_kernel(const float* a, const float* b, float* out, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = a[idx] + b[idx];
    }
}


void launch_add_kernel_float(const float* a, const float* b, float* out, size_t size) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a, b, out, size);
}
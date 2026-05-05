#include <cuda_runtime.h>
#include <cfloat>

static constexpr int Br = 4;   
static constexpr int Bc = 32;  // keys per tile  (= warp size)

// Q/K/V/O: flat pointer to (B*H, seq, d) — blockIdx.y selects the batch-head slice
__global__ void flash_attn_v2_kernel(const float* __restrict__ Q,
                                     const float* __restrict__ K,
                                     const float* __restrict__ V,
                                     float* __restrict__ O,
                                     int M, int N, int d) {
    int bh = blockIdx.y;
    Q += bh * M * d;
    K += bh * N * d;
    V += bh * N * d;
    O += bh * M * d;

    float scale   = rsqrtf((float)d);
    int qi_start  = blockIdx.x * Br;
    if (qi_start >= M) return;
    int actual_Br = min(Br, M - qi_start);

    extern __shared__ float smem[];
    float* sq = smem;           // [Br * d]
    float* sk = sq + Br * d;    // [Bc * d]
    float* sv = sk + Bc * d;    // [Bc * d]
    float* ss = sv + Bc * d;    // [Br * Bc]
    float* sO = ss + Br * Bc;   // [Br * d]

    int tid = threadIdx.x;
    int row = tid / Bc;
    int col = tid % Bc;

    // G->S:tile Q 
    for (int t = tid; t < Br * d; t += blockDim.x) {
        int r = t / d;
        int c = t % d;
        sq[t] = (qi_start + r < M) ? Q[(qi_start + r) * d + c] : 0.0f;
        sO[t] = 0.0f;
    }
    __syncthreads();

    // 前面的 mi li
    float mi = -FLT_MAX;
    float li = 0.0f;

    for (int j = 0; j < N; j += Bc) {
        int chunk = min(Bc, N - j);

        // G->S:tile K, V
        for (int t = tid; t < Bc * d; t += blockDim.x) {
            int r = t / d;
            int c = t % d;
            sk[t] = (j + r < N) ? K[(j + r) * d + c] : 0.0f;
            sv[t] = (j + r < N) ? V[(j + r) * d + c] : 0.0f;
        }
        __syncthreads();

        // GEMM:  Sij[row][col] = dot(sq[row], sk[col]) * scale
        {
            float s = -FLT_MAX;
            if (row < actual_Br && col < chunk) {
                s = 0.0f;
                for (int i = 0; i < d; ++i)
                    s += sq[row * d + i] * sk[col * d + i];
                s *= scale;
            }
            ss[row * Bc + col] = s;
        }
        __syncthreads();

        // 更新mi li 并指数化ss
        float rescale;
        {
            float val = ss[row * Bc + col];
            for (int offset = 16; offset > 0; offset >>= 1)
                val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset));

            float m_new = fmaxf(mi, val);
            rescale = expf(mi - m_new);

            float sp = (col < chunk) ? expf(ss[row * Bc + col] - m_new) : 0.0f;
            ss[row * Bc + col] = sp;

            float l_sum = sp;
            for (int offset = 16; offset > 0; offset >>= 1)
                l_sum += __shfl_xor_sync(0xffffffff, l_sum, offset);

            mi = m_new;
            li = li * rescale + l_sum;
        }
        __syncthreads();
        // GEMM: ss * sv
        if (row < actual_Br) {
            for (int k = col; k < d; k += Bc) {
                float acc = sO[row * d + k] * rescale;
                for (int t = 0; t < chunk; ++t)
                    acc += ss[row * Bc + t] * sv[t * d + k];
                sO[row * d + k] = acc;
            }
        }
        __syncthreads();
    }
    // S->G: tile O
    if (row < actual_Br) {
        float inv_l = 1.0f / li;
        for (int k = col; k < d; k += Bc)
            O[(qi_start + row) * d + k] = sO[row * d + k] * inv_l;
    }
}

// B*H batched launch: Q/K/V/O are contiguous (B, H, seq, d) tensors flattened to (B*H, seq, d)
void run_flash_attn_v2(const float* Q, const float* K, const float* V,
                       float* O, int B, int H, int M, int N, int d,
                       cudaStream_t stream) {
    // SMEM: sq[Br*d] + sk[Bc*d] + sv[Bc*d] + ss[Br*Bc] + sO[Br*d]
    size_t smem = (2 * Br * d + 2 * Bc * d + Br * Bc) * sizeof(float);

    dim3 grid((M + Br - 1) / Br, B * H);
    int  block_size = Br * Bc;

    flash_attn_v2_kernel<<<grid, block_size, smem, stream>>>(Q, K, V, O, M, N, d);
}

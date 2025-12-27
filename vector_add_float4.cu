__global__ void vector_add(const float* A,
                           const float* B,
                           float* C,
                           int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    int N4 = N / 4;

    // Process pairs
    for (int idx = tid; idx < N / 4; idx += stride) {
        float4 x = reinterpret_cast<const float4*>(A)[idx];
        float4 y = reinterpret_cast<const float4*>(B)[idx];

        float4 z;
        z.x = x.x + y.x;
        z.y = x.y + y.y;
        z.z = x.z + y.z;
        z.w = x.w + y.w; 
        reinterpret_cast<float4*>(C)[idx] = z;
    }
    
    if (tid == 0) {
        for (int i = N4 * 4; i < N; i++) {
            C[i] = A[i] + B[i];
        }
    }

}

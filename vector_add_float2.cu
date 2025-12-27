__global__ void vector_add(const float* A,
                           const float* B,
                           float* C,
                           int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    // Process pairs
    for (int idx = tid; idx < N / 2; idx += stride) {
        float2 x = reinterpret_cast<const float2*>(A)[idx];
        float2 y = reinterpret_cast<const float2*>(B)[idx];

        float2 z;
        z.x = x.x + y.x;
        z.y = x.y + y.y;

        reinterpret_cast<float2*>(C)[idx] = z;
    }

    // Handle odd tail element (single thread only)
    if ((N & 1) && tid == 0) {
        C[N - 1] = A[N - 1] + B[N - 1];
    }
}

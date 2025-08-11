#include "util.cu"

__global__ void argmax_kernel(
    const size_t batch_size,
    const size_t vec_size,
    const float* in,
    float* out)
{
    const size_t batch_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (batch_idx >= batch_size)
        return;

    const float* vec = in + batch_idx * vec_size;
    int max_idx = 0;
    float max_val = vec[0];
    for (int i = 1; i < vec_size; i++) {
        if (vec[i] > max_val) {
            max_val = vec[i];
            max_idx = i;
        }
    }
    float* out_vec = out + batch_idx * vec_size;
    for (int i = 0; i < vec_size; i++) {
        out_vec[i] = 0.0F;
    }
    out_vec[max_idx] = 1.0F;
}

extern "C" void argmax(const size_t vec_size, const float* in, float* out)
{
    const size_t blocks = (vec_size + threadsPerBlock - 1) / threadsPerBlock;
    argmax_kernel<<<blocks, threadsPerBlock>>>(1, vec_size, in, out);
}

__global__ void argmax_backward_kernel(
    const size_t batch_size,
    const size_t vec_size,
    const float* output_grad,
    float* input_grad)
{
    const size_t batch_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (batch_idx >= batch_size)
        return;

    const float* grad_in = output_grad + batch_idx * vec_size;
    float* grad_out = input_grad + batch_idx * vec_size;
    for (int i = 0; i < vec_size; i++) {
        grad_out[i] = grad_in[i];
    }
}

extern "C" void argmax_backward(const size_t vec_size, const float* output_grad, float* input_grad)
{
    const size_t blocks = (vec_size + threadsPerBlock - 1) / threadsPerBlock;
    argmax_backward_kernel<<<blocks, threadsPerBlock>>>(1, vec_size, output_grad, input_grad);
}
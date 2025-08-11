extern "C" __global__ void kernel(
    const int single_size,
    const int batch_size,
    const float* input,
    float* output)
{
    const int loc_in_batch = blockIdx.x * blockDim.x + threadIdx.x;
    if (loc_in_batch >= batch_size) return;

    const float* this_elem = input + loc_in_batch * single_size;
    float* this_output = output + loc_in_batch * single_size;

    // Zero-init output
    for (int i = 0; i < single_size; i++) {
        this_output[i] = 0.0f;
    }

    // Top2
    int idx1 = -1, idx2 = -1;
    const float NEG_INF = -3.402823466e+38f;
    float val1 = NEG_INF, val2 = NEG_INF;

    for (int i = 0; i < single_size; i++) {
        float v = this_elem[i];
        if (v > val1) {
            val2 = val1;
            idx2 = idx1;
            val1 = v;
            idx1 = i;
        } else if (v > val2) {
            val2 = v;
            idx2 = i;
        }
    }

    // Softmax
    float max_val = (val1 > val2) ? val1 : val2;
    float exp1 = expf(val1 - max_val);
    float exp2 = expf(val2 - max_val);
    float sum_exp = exp1 + exp2;

    this_output[idx1] = exp1 / sum_exp;
    this_output[idx2] = exp2 / sum_exp;
}
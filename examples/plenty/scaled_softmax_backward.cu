__device__ float sigmoid(float in) {
    return 1.0F / (1.0F + expf(-in));
}

extern "C" __global__ void kernel(
    const int single_size,
    const int batch_size,
    const float k,
    const float* input,
    float* output)
{
    const int loc_in_batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (loc_in_batch >= batch_size)
    {
        return;
    }

    const float* this_elem = input + loc_in_batch * single_size;
    float* this_output = output + loc_in_batch * single_size;

    for (int loc_in_elem = 0; loc_in_elem < single_size; loc_in_elem++)
    {
        this_output[loc_in_elem] = 2 * this_elem[loc_in_elem];
        // this_output[loc_in_elem] = (k * sigmoid(k * (this_elem[loc_in_elem] - 0.5F)) * (1 - sigmoid(k * (this_elem[loc_in_elem] - 0.5F)))) / (sigmoid(0.5F * k) - sigmoid(-0.5F * k));
    }
}
extern "C" __global__ void kernel(
    const int single_size,
    const int batch_size,
    const float* input,
    float* output)
{
    const int loc_in_batch = blockIdx.x * blockDim.x + threadIdx.x;

    if (loc_in_batch >= batch_size)
    {
        return;
    }

    const float* this_elem = input + loc_in_batch * single_size;

    float max = this_elem[0];
    int argmax = 0;

    for (int loc_in_elem = 1; loc_in_elem < single_size; loc_in_elem++)
    {
        const float val = this_elem[loc_in_elem];
        if (val > max)
        {
            max = val;
            argmax = loc_in_elem;
        }
    }

    float* this_output = output + loc_in_batch * single_size;

    for (int loc_in_elem = 0; loc_in_elem < single_size; loc_in_elem++)
    {
        this_output[loc_in_elem] = (loc_in_elem == argmax) ? 1.0F : 0.0F;
    }
}
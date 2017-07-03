__global__ void calcNeurons(
        float t, unsigned layer_size,
        unsigned *spike_count, unsigned *spikes,
        float *spike_time, bool *fired)
{
    unsigned id = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (id < layer_size) {
        __shared__ unsigned spikes_block[BLOCK_SIZE];
        __shared__ volatile unsigned spike_count_block;
        __shared__ volatile unsigned spikes_idx;

        if (threadIdx.x == 0) {
            spike_count_block = 0;
        }
        __syncthreads();

        ////////////////////////////////////////////////////////////
        // begin
        ////////////////////////////////////////////////////////////
        if (t >= spike_time[id] && !fired[id]) {
            fired[id] = true;
            spikes_block[atomicAdd((unsigned *)&spike_count_block, 1)] = id;
        }
        ////////////////////////////////////////////////////////////
        // end
        ////////////////////////////////////////////////////////////

        __syncthreads();
        if (threadIdx.x == 0) {
            if (spike_count_block > 0) {
                spikes_idx = atomicAdd(&spike_count[0], spike_count_block);
            }
        }

        __syncthreads();
        if (threadIdx.x < spike_count_block) {
            spikes[spikes_idx + threadIdx.x] = spikes_block[threadIdx.x];
        }
    }
}

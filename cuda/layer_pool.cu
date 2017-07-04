#include <stdio.h>

__global__ void calcNeurons(
        float t, unsigned layer_size,
        unsigned *spike_count, unsigned *spikes, float *in_syn,
        float *V, bool *fired,
        unsigned threshold)
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

        V[id] += in_syn[id];
        in_syn[id] = 0;
        bool fire = false;

        ////////////////////////////////////////////////////////////
        // begin
        ////////////////////////////////////////////////////////////
        if (V[id] >= threshold && !fired[id]) {
            fired[id] = true;
            fire = true;
        }
        ////////////////////////////////////////////////////////////
        // end
        ////////////////////////////////////////////////////////////

        if (fire)
            spikes_block[atomicAdd((unsigned *)&spike_count_block, 1)] = id;

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


__global__ void calcSynapses(
        float t, unsigned layer_size_post,
        unsigned *spike_count_pre, unsigned *spikes_pre, float *in_syn_post,
        int *g)
{
    unsigned id_post = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (id_post < layer_size_post) {
        float lin_syn_post = in_syn_post[id_post];

        for (unsigned i = 0; i < spike_count_pre[0]; i++) {

            ////////////////////////////////////////////////////////////
            // begin
            ////////////////////////////////////////////////////////////
            lin_syn_post += g[spikes_pre[i] * layer_size_post + id_post];
            ////////////////////////////////////////////////////////////
            // end
            ////////////////////////////////////////////////////////////
        }

        in_syn_post[id_post] = lin_syn_post;
    }
}

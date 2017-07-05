#include <stdio.h>

__global__ void calcNeurons(
        float t, unsigned layer_size,
        unsigned *spike_count, unsigned *spikes, float *in_syn,
        float *V, bool *fired, bool *allow_fire_loc,
        unsigned threshold, unsigned map_size)
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
        if (V[id] > threshold && allow_fire_loc[id % map_size]) {
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
        int *g, float *weights)
{
    unsigned id_post = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (id_post < layer_size_post) {
        float lin_syn_post = in_syn_post[id_post];

        for (unsigned i = 0; i < spike_count_pre[0]; i++) {

            ////////////////////////////////////////////////////////////
            // begin
            ////////////////////////////////////////////////////////////
            int lg = g[spikes_pre[i] * layer_size_post + id_post];
            if (lg > -1) {
                lin_syn_post += *(weights + lg);
            }
            ////////////////////////////////////////////////////////////
            // end
            ////////////////////////////////////////////////////////////
        }

        in_syn_post[id_post] = lin_syn_post;
    }
}


__global__ void learnSynapsesPost(
        float t, unsigned layer_size_pre, unsigned layer_size_post,
        unsigned *spike_count_post, unsigned *spikes_post, bool *fired_pre,
        int *g, float *weights, int *winners_intramap, bool *plastic,
        float a_plus, float a_minus, unsigned map_size)
{
    unsigned id_pre = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    __shared__ unsigned shared_spikes[BLOCK_SIZE];

    if (id_pre < layer_size_pre) {
        unsigned lspike_count_post = spike_count_post[0];
        unsigned num_spike_subsets = (lspike_count_post + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int r = 0; r < num_spike_subsets; r++) {
            unsigned lmax;
            if (r == num_spike_subsets - 1)
                lmax = (lspike_count_post - 1) % BLOCK_SIZE + 1;
            else
                lmax = BLOCK_SIZE;
            if (threadIdx.x < lmax) {
                shared_spikes[threadIdx.x] = spikes_post[r * BLOCK_SIZE + threadIdx.x];
            }
            __syncthreads();
            for (int j = 0; j < lmax; j++) {

                ////////////////////////////////////////////////////////////
                // begin
                ////////////////////////////////////////////////////////////
                if (g[id_pre * layer_size_post + shared_spikes[j]] > -1 && *(plastic)) {
                    float *pw = weights + g[id_pre * layer_size_post + shared_spikes[j]];
                    float w = *pw;

                    unsigned id_post = shared_spikes[j];
                    if (winners_intramap[id_post / map_size] == id_post) { // if post-neuron is winner of current map
                        if (fired_pre[id_pre])
                            *pw += a_plus * w * (1 - w);
                        else
                            *pw -= a_minus * w * (1 - w);
                    }
                }
                ////////////////////////////////////////////////////////////
                // end
                ////////////////////////////////////////////////////////////
            }
        }
    }
}

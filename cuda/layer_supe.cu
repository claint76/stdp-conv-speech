#include <stdio.h>

__global__ void calcNeurons(
        float t, int layer_size,
        int *spike_count, int *spikes, float *in_syn,
        float *V, bool *fired,
        int threshold)
{
    int id = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (id < layer_size) {
        __shared__ int spikes_block[BLOCK_SIZE];
        __shared__ volatile int spike_count_block;
        __shared__ volatile int spikes_idx;

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
        if (V[id] > threshold && !fired[id]) {
            fired[id] = true;
            fire = true;
        }
        ////////////////////////////////////////////////////////////
        // end
        ////////////////////////////////////////////////////////////

        if (fire)
            spikes_block[atomicAdd((int *)&spike_count_block, 1)] = id;

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
        float t, int layer_size_post,
        int *spike_count_pre, int *spikes_pre, float *in_syn_post,
        int *g, float *weights)
{
    int id_post = BLOCK_SIZE * blockIdx.x + threadIdx.x;

    if (id_post < layer_size_post) {
        float lin_syn_post = in_syn_post[id_post];

        for (int i = 0; i < spike_count_pre[0]; i++) {

            ////////////////////////////////////////////////////////////
            // begin
            ////////////////////////////////////////////////////////////
            int lg = g[spikes_pre[i] * layer_size_post + id_post];
            lin_syn_post += *(weights + lg);
            ////////////////////////////////////////////////////////////
            // end
            ////////////////////////////////////////////////////////////
        }

        in_syn_post[id_post] = lin_syn_post;
    }
}


__global__ void learnSynapsesPost(
        float t, int layer_size_pre, int layer_size_post,
        int *spike_count_post, int *spikes_post, bool *fired_pre,
        int *g, float *weights, bool *plastic, int *label,
        float a_plus, float a_minus)
{
    int id_pre = BLOCK_SIZE * blockIdx.x + threadIdx.x;
    __shared__ int shared_spikes[BLOCK_SIZE];

    if (id_pre < layer_size_pre) {
        int lspike_count_post = spike_count_post[0];
        int num_spike_subsets = (lspike_count_post + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int r = 0; r < num_spike_subsets; r++) {
            int lmax;
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
                if (*plastic) {
                    int id_post = shared_spikes[j];

                    float *pw = weights + g[id_pre * layer_size_post + id_post];
                    float w = *pw;

                    if (fired_pre[id_pre]) {
                        if (id_post == *label) {
                            *pw += a_plus * w * (1 - w);
                        } else {
                            *pw -= a_minus * w * (1 - w);
                        }
                    } else {
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

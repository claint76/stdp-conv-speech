__global__ void get_intermap_firing_winners(
        unsigned *glbSpk, unsigned *glbSpkCnt, float *V,
        int *winners_intermap, float *winnersV_intermap, int *mutex,
        unsigned map_size)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < glbSpkCnt[0]) {
        unsigned id = glbSpk[tid];

        bool need_lock = true;
        while (need_lock) {
            if (atomicCAS(mutex, 0, 1) == 0) {
                // critical section
                if (V[id] > winnersV_intermap[id % map_size]) {
                    winnersV_intermap[id % map_size] = V[id];
                    winners_intermap[id % map_size] = id;
                }
                // end critical section
                atomicExch(mutex, 0);
                need_lock = false;
            }
        }
    }
}


__global__ void clean_spikes(
        unsigned *glbSpk, unsigned *glbSpkCnt, float *V, bool *fired,
        int *winners_intermap, bool *allow_fire_loc, int *mutex, unsigned *spikes_temp, unsigned *spike_count_temp,
        unsigned map_size)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < glbSpkCnt[0]) {
        unsigned id = glbSpk[tid];

        if (winners_intermap[id % map_size] == id) {
            unsigned spike_id = atomicAdd(spike_count_temp, 1);
            spikes_temp[spike_id] = id;
            fired[id] = true;
            allow_fire_loc[id % map_size] = false;
        }
    }
}


// allow fired neuron with highest potential in a map to do stdp
__global__ void get_intramap_stdp_winners(
        unsigned *glbSpk, unsigned *glbSpkCnt, float *V,
        int *winners_intramap, float *winnersV_intramap, bool *allow_stdp_map, bool *allow_stdp_loc, int *mutex,
        unsigned map_size)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < glbSpkCnt[0]) {
        unsigned id = glbSpk[tid];

        if (allow_stdp_map[id / map_size] && allow_stdp_loc[id % map_size]) {
            bool need_lock = true;
            while (need_lock) {
                if (atomicCAS(mutex, 0, 1) == 0) {
                    // critical section
                    if (V[id] > winnersV_intramap[id / map_size]) {
                        winnersV_intramap[id / map_size] = V[id];
                        winners_intramap[id / map_size] = id;
                    }
                    // end critical section
                    atomicExch(mutex, 0);
                    need_lock = false;
                }
            }
        }
    }
}


// eliminate winners which are near to a stronger winner in other maps
__global__ void get_intermap_stdp_winners(
        int *winners_intramap, float *winnersV_intramap,
        unsigned map_num, unsigned map_size, unsigned width, unsigned radius)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < map_num) {
        int map = tid;

        if (winners_intramap[map] != -1) {
            int id = winners_intramap[map];
            int r = id % map_size / width;
            int c = id % map_size % width;
            int l = radius;

            for (int i = 0; i < map_num; i++) {
                int id2 = winners_intramap[i];
                int r2 = id2 % map_size / width;
                int c2 = id2 % map_size % width;

                if (map != i && winnersV_intramap[map] < winnersV_intramap[i]
                        && r >= r2 - l && r <= r2 + l && c >= c2 - l && c <= c2 + l) {
                    winners_intramap[map] = -1;
                    winnersV_intramap[map] = 0;
                    break;
                }
            }
        }
    }
}


// set allow_stdp_map and allow_stdp_loc
__global__ void disallow_nearby_stdp(
        int *winners_intramap, bool *allow_stdp_map, bool *allow_stdp_loc,
        unsigned map_num, unsigned map_size, unsigned width, unsigned height, unsigned radius)
{
    unsigned tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < map_num) {
        int map = tid;

        if (winners_intramap[map] != -1) {
            int id = winners_intramap[map];
            int r = id % map_size / width;
            int c = id % map_size % width;
            int l = radius;

            allow_stdp_map[map] = false;

            for (int i = (r-l < 0 ? 0 : r-l); i <= (r+l > height-1 ? height-1 : r+l); i++)
                for (int j = (c-l < 0 ? 0 : c-l); j <= (c+l > width-1 ? width-1 : c+l); j++)
                    allow_stdp_loc[i * width + j] = false;
        }
    }
}

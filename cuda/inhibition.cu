__global__ void get_intermap_firing_winners(
        int *glbSpk, int *glbSpkCnt, float *V,
        int *winners_intermap, float *winnersV_intermap, int *mutex,
        int map_size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < glbSpkCnt[0]) {
        int id = glbSpk[tid];

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
        int *glbSpk, int *glbSpkCnt, float *V, bool *fired,
        int *winners_intermap, bool *allow_fire_loc, int *mutex, int *spikes_temp, int *spike_count_temp,
        int map_size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < glbSpkCnt[0]) {
        int id = glbSpk[tid];

        if (winners_intermap[id % map_size] == id) {
            int spike_id = atomicAdd(spike_count_temp, 1);
            spikes_temp[spike_id] = id;
            fired[id] = true;
            allow_fire_loc[id % map_size] = false;
        }
    }
}


// allow fired neuron with highest potential in a map to do stdp
__global__ void get_intramap_stdp_winners(
        int *glbSpk, int *glbSpkCnt, float *V,
        int *winners_intramap, float *winnersV_intramap, bool *allow_stdp_map, bool *allow_stdp_loc, int *mutex,
        int map_num, int map_size, int width, int sec_num, int sec_size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < glbSpkCnt[0]) {
        int id = glbSpk[tid];
        int sec = id % map_size / width / sec_size;
        int map = id / map_size;

        if (allow_stdp_map[sec * map_num + map] && allow_stdp_loc[id % map_size]) {
            bool need_lock = true;
            while (need_lock) {
                if (atomicCAS(mutex, 0, 1) == 0) {
                    // critical section
                    if (V[id] > winnersV_intramap[sec * map_num + map]) {
                        winnersV_intramap[sec * map_num + map] = V[id];
                        winners_intramap[sec * map_num + map] = id;
                    }
                    // end critical section
                    atomicExch(mutex, 0);
                    need_lock = false;
                }
            }
        }
    }
}


// set allow_stdp_map and allow_stdp_loc
__global__ void disallow_nearby_stdp(
        int *winners_intramap, bool *allow_stdp_map, bool *allow_stdp_loc,
        int map_num, int map_size, int width, int sec_num, int sec_size, int radius)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < sec_num * map_num) {
        int sec = tid / map_num;
        int map = tid % map_num;

        if (winners_intramap[sec * map_num + map] != -1) {
            int id = winners_intramap[sec * map_num + map];
            int r = id % map_size / width;
            int c = id % map_size % width;
            int l = radius;

            allow_stdp_map[sec * map_num + map] = false;

            for (int i = (r-l < sec*sec_size ? sec*sec_size : r-l); i <= (r+l > (sec+1)*sec_size-1 ? (sec+1)*sec_size-1 : r+l); i++)
                for (int j = (c-l < 0 ? 0 : c-l); j <= (c+l > width-1 ? width-1 : c+l); j++)
                    allow_stdp_loc[i * width + j] = false;
        }
    }
}

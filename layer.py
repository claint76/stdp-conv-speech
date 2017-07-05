#!/usr/bin/env python

import numpy as np

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule


block_size = 64


def get_kernels(cu_file, funcs):
    with open('cuda/' + cu_file) as f:
        source = f.read()
    source = source.replace('BLOCK_SIZE', str(block_size))
    mod = SourceModule(source)
    return tuple(mod.get_function(func) for func in funcs)


class LayerBase:
    def __init__(self, width, height, map_num):
        self.width = np.uint32(width)
        self.height = np.uint32(height)
        self.map_size = np.uint32(width * height)
        self.map_num = np.uint32(map_num)
        self.layer_size = np.uint32(self.map_num * self.map_size)

        self.spike_count = gpuarray.empty(shape=(1,), dtype=np.uint32)
        self.spikes = gpuarray.empty(shape=(self.layer_size,), dtype=np.uint32)
        self.fired = gpuarray.empty(shape=(self.layer_size,), dtype=np.bool)

    def reset(self):
        self.spike_count.fill(0)
        self.fired.fill(False)


class LayerInput(LayerBase):
    (calc_neurons,) = get_kernels('layer_input.cu', ['calcNeurons'])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.spike_time = gpuarray.empty(shape=(self.layer_size,), dtype=np.float32) # no need to reset

        self.reset()

    def step(self, t):
        grid_size = int((self.layer_size + block_size - 1) // block_size) # must be converted to int
        self.spike_count.fill(0)
        self.calc_neurons(
                t, self.layer_size,
                self.spike_count, self.spikes,
                self.spike_time, self.fired,
                block=(block_size, 1, 1), grid=(int(grid_size), 1))


class LayerNonInput(LayerBase):
    def __init__(self, layer_pre, win_width, win_height, stride, map_num, threshold):
        self.layer_pre = layer_pre
        self.win_width = np.uint32(win_width)
        self.win_height = np.uint32(win_height)
        self.win_size = np.uint32(win_width * win_height)
        self.stride = np.uint32(stride)
        self.threshold = np.uint32(threshold)

        width = (layer_pre.width - win_width) // stride + 1
        height = (layer_pre.height - win_height) // stride + 1
        super().__init__(width, height, map_num)

        self.V = gpuarray.empty(shape=(self.layer_size,), dtype=np.float32)
        self.in_syn = gpuarray.zeros(shape=(self.layer_size,), dtype=np.float32)

    def reset(self):
        super().reset()
        self.V.fill(0)


class LayerConv(LayerNonInput):
    calc_neurons, calc_synapses, learn_synapses_post = \
            get_kernels('layer_conv.cu', ['calcNeurons', 'calcSynapses', 'learnSynapsesPost'])

    get_intermap_firing_winners, clean_spikes, disallow_nearby_stdp, get_intramap_stdp_winners, get_intermap_stdp_winners = \
            get_kernels('inhibition.cu', ['get_intermap_firing_winners', 'clean_spikes', 'disallow_nearby_stdp', 'get_intramap_stdp_winners', 'get_intermap_stdp_winners'])

    def __init__(self, layer_pre, win_width, win_height, stride, map_num, threshold, a_plus, a_minus, learning_rounds):
        super().__init__(layer_pre, win_width, win_height, stride, map_num, threshold)
        self.a_plus = np.float32(a_plus)
        self.a_minus = np.float32(a_minus)
        self.learning_rounds = learning_rounds

        self.plastic = gpuarray.zeros(shape=(1,), dtype=np.bool)
        self.weights = gpuarray.to_gpu(np.random.normal(0.8, 0.01, (self.map_num * self.win_size * self.layer_pre.map_num,)).astype(np.float32))
        self.g = gpuarray.empty(shape=(self.layer_size * self.layer_pre.layer_size,), dtype=np.int32)

        self.winners_intermap = gpuarray.empty(shape=(self.map_size,), dtype=np.int32) # inhibit other firing, type should be compatible with atomicCAS
        self.winners_intramap = gpuarray.empty(shape=(self.map_num,), dtype=np.int32)
        self.winnersV_intermap = gpuarray.empty(shape=(self.map_size,), dtype=np.float32)
        self.winnersV_intramap = gpuarray.empty(shape=(self.map_num,), dtype=np.float32)
        self.spikes_temp = gpuarray.empty(shape=(self.map_size,), dtype=np.int32)
        self.spike_count_temp = gpuarray.empty(shape=(1,), dtype=np.int32)
        self.mutex = gpuarray.empty(shape=(1,), dtype=np.int32)
        self.allow_fire_loc = gpuarray.empty(shape=(self.map_size,), dtype=np.bool) # inhibit other firing on same location of other maps in the following timesteps
        self.allow_stdp_map = gpuarray.empty(shape=(self.map_num,), dtype=np.bool) # inhibit STDP on same map in the following timesteps
        self.allow_stdp_loc = gpuarray.empty(shape=(self.map_size,), dtype=np.bool) # inhibit STDP on other maps in the following timesteps

        self.generate_connections()
        self.reset()

    def reset(self):
        super().reset()
        self.winners_intermap.fill(-1)
        self.winners_intramap.fill(-1)
        self.winnersV_intermap.fill(0)
        self.winnersV_intramap.fill(0)
        self.spike_count_temp.fill(0)
        self.mutex.fill(0)
        self.allow_fire_loc.fill(True)
        self.allow_stdp_map.fill(True)
        self.allow_stdp_loc.fill(True)

    def generate_connections(self):
        g_host = self.g.get()
        g_host.fill(-1)
        for ipost in range(self.map_size):
            rpost = ipost // self.width
            cpost = ipost % self.width

            # start point of input window of current post-neuron
            rpre_base = rpost * self.stride
            cpre_base = cpost * self.stride

            for i in range(self.win_size):
                rpre = rpre_base + i // self.win_width
                cpre = cpre_base + i % self.win_width
                ipre = rpre * self.layer_pre.width + cpre

                for map_post in range(self.map_num):
                    for map_pre in range(self.layer_pre.map_num):
                        nid_pre = map_pre * self.layer_pre.map_size + ipre
                        gid = nid_pre * self.layer_size + map_post * self.map_size + ipost # index of current synapse
                        g_host[gid] = map_post * self.layer_pre.map_num * self.win_size + map_pre * self.win_size + i
        self.g.set(g_host)


    def step(self, t):
        grid_size = int((self.layer_size + block_size - 1) // block_size) # must be converted to int
        self.calc_synapses(
                t, self.layer_size,
                self.layer_pre.spike_count, self.layer_pre.spikes, self.in_syn,
                self.g, self.weights,
                block=(block_size,1,1), grid=(grid_size,1))

        grid_size = int((self.layer_pre.layer_size + block_size - 1) // block_size) # must be converted to int
        self.learn_synapses_post(
                t, self.layer_pre.layer_size, self.layer_size,
                self.spike_count, self.spikes, self.layer_pre.fired,
                self.g, self.weights, self.winners_intramap, self.plastic,
                self.a_plus, self.a_minus, self.map_size,
                block=(block_size,1,1), grid=(grid_size,1))

        self.spike_count.fill(0)
        grid_size = int((self.layer_size + block_size - 1) // block_size) # must be converted to int
        self.calc_neurons(
                t, self.layer_size,
                self.spike_count, self.spikes, self.in_syn,
                self.V, self.fired, self.allow_fire_loc,
                self.threshold, self.map_size,
                block=(block_size,1,1), grid=(grid_size,1))


    def inhibit(self):
        # neuron inhibition
        grid_size = int((self.spike_count.get()[0] + block_size - 1) // block_size)
        if grid_size == 0:
            return
        self.get_intermap_firing_winners(
                self.spikes, self.spike_count, self.V,
                self.winners_intermap, self.winnersV_intermap, self.mutex,
                self.map_size,
                block=(block_size,1,1), grid=(grid_size,1))

        self.spike_count_temp.fill(0)
        self.clean_spikes(
                self.spikes, self.spike_count, self.V, self.fired,
                self.winners_intermap, self.allow_fire_loc, self.mutex, self.spikes_temp, self.spike_count_temp,
                self.map_size,
                block=(block_size,1,1), grid=(grid_size,1))
        cuda.memcpy_dtod(self.spikes.gpudata, self.spikes_temp.gpudata, self.spikes_temp.nbytes)
        cuda.memcpy_dtod(self.spike_count.gpudata, self.spike_count_temp.gpudata, self.spike_count_temp.nbytes)

        # stdp inhibition
        grid_size = int((self.map_num + block_size - 1) // block_size)
        self.disallow_nearby_stdp(
                self.winners_intramap, self.allow_stdp_map, self.allow_stdp_loc,
                self.map_num, self.map_size, self.width, self.height, self.win_width//2, # must be called before winners(V)_intramap are reset
                block=(block_size,1,1), grid=(grid_size,1))

        self.winners_intramap.fill(-1)
        self.winnersV_intramap.fill(0)

        grid_size = int((self.spike_count.get()[0] + block_size - 1) // block_size)
        self.get_intramap_stdp_winners(
                self.spikes, self.spike_count, self.V,
                self.winners_intramap, self.winnersV_intramap, self.allow_stdp_map, self.allow_stdp_loc, self.mutex,
                self.map_size,
                block=(block_size,1,1), grid=(grid_size,1))
        grid_size = int((self.map_num + block_size - 1) // block_size)
        self.get_intermap_stdp_winners(
                self.winners_intramap, self.winnersV_intramap,
                self.map_num, self.map_size, self.width, self.win_width//2,
                block=(block_size,1,1), grid=(grid_size,1))


class LayerPool(LayerNonInput):
    calc_neurons, calc_synapses = get_kernels('layer_pool.cu', ['calcNeurons', 'calcSynapses'])

    def __init__(self, layer_pre, win_width, win_height, stride):
        super().__init__(layer_pre, win_width, win_height, stride, map_num=layer_pre.map_num, threshold=0)

        self.g = gpuarray.empty(shape=(self.layer_size * self.layer_pre.layer_size,), dtype=np.bool)

        self.generate_connections()
        self.reset()

    def generate_connections(self):
        g_host = self.g.get()
        g_host.fill(0)
        for ipost in range(self.map_size):
            rpost = ipost // self.width
            cpost = ipost % self.width

            # start point of input window of current post-neuron
            rpre_base = rpost * self.stride
            cpre_base = cpost * self.stride

            for i in range(self.win_size):
                rpre = rpre_base + i // self.win_width
                cpre = cpre_base + i % self.win_width
                ipre = rpre * self.layer_pre.width + cpre

                for map_post in range(self.map_num):
                    map_pre = map_post
                    nid_pre = map_pre * self.layer_pre.map_size + ipre
                    gid = nid_pre * self.layer_size + map_post * self.map_size + ipost # index of current synapse
                    g_host[gid] = 1
        self.g.set(g_host)

    def step(self, t):
        grid_size = int((self.layer_size + block_size - 1) // block_size) # must be converted to int
        self.calc_synapses(
                t, self.layer_size,
                self.layer_pre.spike_count, self.layer_pre.spikes, self.in_syn,
                self.g,
                block=(block_size,1,1), grid=(grid_size,1))

        self.spike_count.fill(0)
        self.calc_neurons(
                t, self.layer_size,
                self.spike_count, self.spikes, self.in_syn,
                self.V, self.fired,
                self.threshold,
                block=(block_size,1,1), grid=(grid_size,1))

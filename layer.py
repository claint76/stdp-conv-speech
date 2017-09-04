#!/usr/bin/env python3

import numpy as np
import pickle
import os.path

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
        self.width = np.int32(width)
        self.height = np.int32(height)
        self.map_size = np.int32(width * height)
        self.map_num = np.int32(map_num)
        self.layer_size = np.int32(self.map_num * self.map_size)

        self.spike_count = gpuarray.empty(shape=(1,), dtype=np.int32)
        self.spikes = gpuarray.empty(shape=(self.layer_size,), dtype=np.int32)
        self.fired = gpuarray.empty(shape=(self.layer_size,), dtype=np.bool)

    def reset(self):
        self.spike_count.fill(0)
        self.spikes.fill(0)
        self.fired.fill(False)

    def step_synapses(self, t):
        pass

    def step_synapses_post(self, t):
        pass


class LayerInput(LayerBase):
    (calc_neurons,) = get_kernels('layer_input.cu', ['calcNeurons'])

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.spike_time = gpuarray.empty(shape=(self.layer_size,), dtype=np.float32) # no need to reset

        self.reset()

    def step_neurons(self, t):
        grid_size = int((self.layer_size + block_size - 1) // block_size) # must be converted to int
        self.spike_count.fill(0)
        self.calc_neurons(
                t, self.layer_size,
                self.spike_count, self.spikes,
                self.spike_time, self.fired,
                block=(block_size, 1, 1), grid=(grid_size, 1))


class LayerNonInput(LayerBase):
    def __init__(self, layer_pre, win, stride, map_num, threshold):
        self.layer_pre = layer_pre

        if type(win) is tuple:
            self.win_width = np.int32(win[0])
            self.win_height = np.int32(win[1])
        else:
            self.win_width = self.win_height = np.int32(win)
        self.win_size = np.int32(self.win_width * self.win_height)

        self.stride = np.int32(stride)
        self.threshold = np.int32(threshold)

        width = (layer_pre.width - self.win_width) // stride + 1
        height = (layer_pre.height - self.win_height) // stride + 1
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

    def __init__(self, layer_pre, win, stride, map_num, threshold, a_plus, a_minus, learning_rounds):
        super().__init__(layer_pre, win, stride, map_num, threshold)
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
        self.spikes_temp.fill(0)
        self.spike_count_temp.fill(0)
        self.mutex.fill(0)
        self.allow_fire_loc.fill(True)
        self.allow_stdp_map.fill(True)
        self.allow_stdp_loc.fill(True)

    def generate_connections(self):
        g_file = '/tmp/g_{}_{}_{}_{}_{}_{}.pickle'.format(self.layer_pre.width, self.layer_pre.height, self.layer_pre.map_num, self.win_width, self.win_height, self.stride)
        if os.path.isfile(g_file):
            with open(g_file, 'rb') as f:
                g_host = pickle.load(f)
                self.g.set(g_host)
                return

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

        with open(g_file, 'wb') as f:
            pickle.dump(g_host, f)

    def step_synapses(self, t):
        grid_size = int((self.layer_size + block_size - 1) // block_size) # must be converted to int
        self.calc_synapses(
                t, self.layer_size,
                self.layer_pre.spike_count, self.layer_pre.spikes, self.in_syn,
                self.g, self.weights,
                block=(block_size,1,1), grid=(grid_size,1))

    def step_synapses_post(self, t):
        grid_size = int((self.layer_pre.layer_size + block_size - 1) // block_size) # must be converted to int
        self.learn_synapses_post(
                t, self.layer_pre.layer_size, self.layer_size,
                self.spike_count, self.spikes, self.layer_pre.fired,
                self.g, self.weights, self.winners_intramap, self.plastic,
                self.a_plus, self.a_minus, self.map_size,
                block=(block_size,1,1), grid=(grid_size,1))

    def step_neurons(self, t):
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
                self.map_num, self.map_size, self.width, self.height, np.int32(self.win_width), # must be called before winners(V)_intramap are reset
                block=(block_size,1,1), grid=(grid_size,1))

        self.winners_intramap.fill(-1)
        self.winnersV_intramap.fill(0)

        grid_size = int((self.spike_count.get()[0] + block_size - 1) // block_size)
        self.get_intramap_stdp_winners(
                self.spikes, self.spike_count, self.V,
                self.winners_intramap, self.winnersV_intramap, self.allow_stdp_map, self.allow_stdp_loc, self.mutex,
                self.map_size,
                block=(block_size,1,1), grid=(grid_size,1))
        # grid_size = int((self.map_num + block_size - 1) // block_size)
        # self.get_intermap_stdp_winners(
        #         self.winners_intramap, self.winnersV_intramap,
        #         self.map_num, self.map_size, self.width, np.int32(self.win_width),
        #         block=(block_size,1,1), grid=(grid_size,1))

        def is_near(a, b, l):
            ra = a % self.map_size // self.width
            ca = a % self.map_size % self.width
            rb = b % self.map_size // self.width
            cb = b % self.map_size % self.width
            return ra >= rb - l and ra <= rb + l and ca >= cb - l and ca <= cb + l

        winners_intramap = self.winners_intramap.get()
        winnersV_intramap = self.winnersV_intramap.get()
        new_winners_intramap = np.full_like(winners_intramap, -1)
        new_winnersV_intramap = np.full_like(winnersV_intramap, 0)

        while True:
            i = np.argmax(winnersV_intramap)
            if winnersV_intramap[i] == 0:
                break
            new_winners_intramap[i] = winners_intramap[i]
            new_winnersV_intramap[i] = winnersV_intramap[i]

            for j in winnersV_intramap.nonzero()[0]:
                if i != j and is_near(winners_intramap[i], winners_intramap[j], self.win_width) and winnersV_intramap[i] > winnersV_intramap[j]:
                    winners_intramap[j] = -1
                    winnersV_intramap[j] = 0
            winners_intramap[i] = -1
            winnersV_intramap[i] = 0

        self.winners_intramap.set(new_winners_intramap)
        self.winnersV_intramap.set(new_winnersV_intramap)


class LayerPool(LayerNonInput):
    calc_neurons, calc_synapses = get_kernels('layer_pool.cu', ['calcNeurons', 'calcSynapses'])

    def __init__(self, layer_pre, win, stride):
        super().__init__(layer_pre, win, stride, map_num=layer_pre.map_num, threshold=0)

        self.g = gpuarray.empty(shape=(self.layer_size * self.layer_pre.layer_size,), dtype=np.bool)

        self.generate_connections()
        self.reset()

    def generate_connections(self):
        g_file = '/tmp/g_{}_{}_{}_{}_{}_{}.pickle'.format(self.layer_pre.width, self.layer_pre.height, self.layer_pre.map_num, self.win_width, self.win_height, self.stride)
        if os.path.isfile(g_file):
            with open(g_file, 'rb') as f:
                g_host = pickle.load(f)
                self.g.set(g_host)
                return

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

        with open(g_file, 'wb') as f:
            pickle.dump(g_host, f)

    def step_synapses(self, t):
        grid_size = int((self.layer_size + block_size - 1) // block_size) # must be converted to int
        self.calc_synapses(
                t, self.layer_size,
                self.layer_pre.spike_count, self.layer_pre.spikes, self.in_syn,
                self.g,
                block=(block_size,1,1), grid=(grid_size,1))

    def step_neurons(self, t):
        self.spike_count.fill(0)
        grid_size = int((self.layer_size + block_size - 1) // block_size) # must be converted to int
        self.calc_neurons(
                t, self.layer_size,
                self.spike_count, self.spikes, self.in_syn,
                self.V, self.fired,
                self.threshold,
                block=(block_size,1,1), grid=(grid_size,1))


class LayerSupe(LayerNonInput):
    calc_neurons, calc_synapses, learn_synapses_post = \
            get_kernels('layer_supe.cu', ['calcNeurons', 'calcSynapses', 'learnSynapsesPost'])

    def __init__(self, layer_pre, map_num, threshold, a_plus, a_minus, learning_rounds):
        super().__init__(layer_pre, (layer_pre.width, layer_pre.height), 1, map_num, threshold)
        self.a_plus = np.float32(a_plus)
        self.a_minus = np.float32(a_minus)
        self.learning_rounds = learning_rounds

        self.plastic = gpuarray.zeros(shape=(1,), dtype=np.bool)
        self.weights = gpuarray.to_gpu(np.random.normal(0.8, 0.01, (self.layer_size * self.layer_pre.layer_size,)).astype(np.float32))
        self.g = gpuarray.to_gpu(np.arange(self.layer_size * self.layer_pre.layer_size).reshape((self.layer_size, self.layer_pre.layer_size)).transpose().astype(np.int32))
        self.label = gpuarray.empty(shape=(1,), dtype=np.int32)

        self.reset()

    def step_synapses(self, t):
        grid_size = int((self.layer_size + block_size - 1) // block_size) # must be converted to int
        self.calc_synapses(
                t, self.layer_size,
                self.layer_pre.spike_count, self.layer_pre.spikes, self.in_syn,
                self.g, self.weights,
                block=(block_size,1,1), grid=(grid_size,1))

    def step_synapses_post(self, t):
        grid_size = int((self.layer_pre.layer_size + block_size - 1) // block_size) # must be converted to int
        self.learn_synapses_post(
                t, self.layer_pre.layer_size, self.layer_size,
                self.spike_count, self.spikes, self.layer_pre.fired,
                self.g, self.weights, self.plastic, self.label,
                self.a_plus, self.a_minus,
                block=(block_size,1,1), grid=(grid_size,1))

    def step_neurons(self, t):
        self.spike_count.fill(0)
        grid_size = int((self.layer_size + block_size - 1) // block_size) # must be converted to int
        self.calc_neurons(
                t, self.layer_size,
                self.spike_count, self.spikes, self.in_syn,
                self.V, self.fired,
                self.threshold,
                block=(block_size,1,1), grid=(grid_size,1))

#!/usr/bin/env python

import numpy

from layer import LayerInput, LayerConv, LayerPool


class Network:
    def __init__(self, params):
        self.it = 0
        self.dt = 0.1

        self.active_layers = []
        self.layers = []
        for layer_param in params['layers']:
            if layer_param['type'] == 'input':
                self.layers.append(LayerInput(
                    layer_param['width'],
                    layer_param['height'],
                    layer_param['map_num'],
                ))
            elif layer_param['type'] == 'conv':
                self.layers.append(LayerConv(
                    self.layers[-1],
                    layer_param['win_width'],
                    layer_param['win_height'],
                    layer_param['stride'],
                    layer_param['map_num'],
                    layer_param['threshold'],
                    layer_param['a_plus'],
                    layer_param['a_minus'],
                    layer_param['learning_rounds'],
                ))
            elif layer_param['type'] == 'pool':
                self.layers.append(LayerPool(
                    self.layers[-1],
                    layer_param['win_width'],
                    layer_param['win_height'],
                    layer_param['stride'],
                ))
            elif layer_param['type'] == 'globalpool':
                self.layers.append(LayerPool(
                    self.layers[-1],
                    self.layers[-1].width,
                    self.layers[-1].height,
                    1, # stride not used
                ))

    def reset(self):
        for layer in self.layers:
            layer.reset()
        self.it = 0

    def step(self):
        t = numpy.float32(self.it * self.dt)
        for layer in self.active_layers:
            layer.step_synapses(t)
        for layer in self.active_layers:
            layer.step_synapses_post(t)
        for layer in self.active_layers:
            layer.step_neurons(t)
        for layer in self.active_layers:
            if hasattr(layer, 'inhibit'):
                layer.inhibit()
        self.it += 1

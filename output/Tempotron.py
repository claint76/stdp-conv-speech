import numpy as np
import matplotlib.pyplot as plt


class Tempotron:
    def __init__(self, synaptic_efficacies, threshold):
        self.threshold = threshold
        self.efficacies = synaptic_efficacies

    def train(self, io_pairs, steps, learning_rate):
        for i in xrange(steps):
            for spike_times, target in np.random.permutation(io_pairs):
                self.adapt_weights(spike_times, target, learning_rate)
        return

    def compute_vmax_tmax(self, spike_times):
        tmax = 0

        spike_contribs = np.array([len(spike_time) != 0 for spike_time in spike_times])
        total_incoming = spike_contribs * self.efficacies
        vmax = total_incoming.sum()

        return vmax, tmax

    def adapt_weights(self, spike_times, target, learning_rate):
        vmax, tmax = self.compute_vmax_tmax(spike_times)

        if target == True:
            if vmax < self.threshold:
                for i, spike_time in enumerate(spike_times):
                    if len(spike_time) > 0:
                        self.efficacies[i] += learning_rate
        else:
            if vmax >= self.threshold:
                for i, spike_time in enumerate(spike_times):
                    if len(spike_time) > 0:
                        self.efficacies[i] -= learning_rate

        self.efficacies = np.clip(self.efficacies, 0, 1)

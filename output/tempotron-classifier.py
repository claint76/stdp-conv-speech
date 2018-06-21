from __future__ import division

from Tempotron import Tempotron
import numpy as np
import pickle
import matplotlib.pyplot as plt
import argparse


with open('output_train_set.pickle.v2', 'rb') as f:
    train_set = pickle.load(f)
with open('output_test_set.pickle.v2', 'rb') as f:
    test_set = pickle.load(f)

n_input = 450
n_output = 10

# parser = argparse.ArgumentParser()
# parser.add_argument('learning_rate', type=float)
# parser.add_argument('round_num', type=int)
# args = parser.parse_args()

# learning_rate = args.learning_rate
# round_num = args.round_num
learning_rate = 0.02
round_num = 5

tempotrons = []
for i in range(n_output):
    np.random.seed(i)
    efficacies = 0.5 + 0.1 * np.random.random(n_input)
    tempotrons.append(Tempotron(efficacies, threshold=15))

for i in range(n_output):
    train_data = []
    print('neuron {}'.format(i))
    for j in range(len(train_set[1])):
        spike_times = (5 - np.expand_dims(train_set[0][j], 1)) * 100
        spike_times = spike_times.tolist()
        spike_times = [time if time[0] != 500.0 else [] for time in spike_times]
        train_data.append((spike_times, i == train_set[1][j]))

    tempotrons[i].train(train_data, round_num, learning_rate=learning_rate)


correct_count = 0
for j in range(len(test_set[1])):
    spike_times = (5 - np.expand_dims(test_set[0][j], 1)) * 100
    spike_times = spike_times.tolist()
    spike_times = [time if time[0] != 500.0 else [] for time in spike_times]
    vmax_list = []
    for i in range(n_output):
        vmax, tmax = tempotrons[i].compute_vmax_tmax(spike_times)
        vmax_list.append(vmax)
    if vmax_list.index(max(vmax_list)) == test_set[1][j]:
        correct_count += 1
print('test_set accuracy = {}'.format(correct_count / len(test_set[1])))

# fig, axes = plt.subplots(1, 1)
# axes.bar(range(450), tempotrons[0].efficacies)
# axes[1].bar(range(450), tempotrons[1].efficacies)
# axes[2].bar(range(450), tempotrons[2].efficacies)
# plt.show()

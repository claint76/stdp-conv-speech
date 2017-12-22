import numpy as np
import matplotlib.pyplot as plt


xl = np.loadtxt('accuracy_local')
xg = np.loadtxt('accuracy_global')
max_maps = 9

plt.figure(figsize=(5.2, 3.2))
plt.errorbar(xl[:max_maps, 0], xl[:max_maps, 1:].mean(axis=1), yerr=np.std(xl[:max_maps, 1:], axis=1, ddof=1), fmt='-o', capsize=3, label='Local weight sharing')
plt.errorbar(xg[:max_maps, 0], xg[:max_maps, 1:].mean(axis=1), yerr=np.std(xg[:max_maps, 1:], axis=1, ddof=1), fmt='-s', capsize=3, label='Global weight sharing')
plt.ylim([90, 100])
plt.xlabel('Number of feature maps')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()

plt.savefig('comparison_local_global.png')
plt.show()

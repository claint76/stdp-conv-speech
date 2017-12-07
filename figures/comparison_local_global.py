import numpy as np
import matplotlib.pyplot as plt


xl = np.loadtxt('accuracy_local')
xg = np.loadtxt('accuracy_global')

plt.figure(figsize=(5.2, 3.2))
plt.plot(xl[:, 0], xl[:, 1:].mean(axis=1), '-o', label='Local weight sharing')
plt.plot(xg[:, 0], xg[:, 1:].mean(axis=1), '-s', label='Global weight sharing')
plt.ylim([90, 100])
plt.xlabel('Number of feature maps')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()

plt.savefig('comparison_local_global.png')
plt.show()

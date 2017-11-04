import numpy as np
import matplotlib.pyplot as plt


x = np.loadtxt('../output/test_output')

plt.figure(figsize=(5.2, 3.2))
plt.plot(x[0:1000:5,0], x[0:1000:5,1])
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.tight_layout()

plt.savefig('accuracy.png')
plt.show()

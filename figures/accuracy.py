import numpy as np
import matplotlib.pyplot as plt


x = np.loadtxt('../output/test_output')

plt.plot(x[:,0], x[:,1])
plt.savefig('accuracy.tiff')
plt.show()

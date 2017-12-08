import numpy as np
import matplotlib.pyplot as plt


x = np.loadtxt('test_accuracies')

plt.figure(figsize=(5.2, 3.1))
plt.plot(x[0:1000:5,0], x[0:1000:5,1], label='SVM accuracy on SNN output')
plt.axhline(y=95, color='orange', linestyle='--', label='SVM accuracy on MFSC features')
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()

plt.savefig('accuracy.png')
plt.show()

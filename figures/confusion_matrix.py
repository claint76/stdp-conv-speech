import numpy as np
import pickle
import matplotlib.pyplot as plt

dataset = 'timit'
# dataset = 'tidigits'

if dataset == 'timit':
    import sys
    sys.path.append('..')
    from readers.timit import wordlist
    classes = wordlist
else:
    classes = list(range(10))

with open('../output/confusion_matrix.pickle', 'rb') as f:
    cm = pickle.load(f)
cm = cm / cm.sum(axis=1)[:, np.newaxis]  # normalize

plt.figure(figsize=(5.2, 3.2))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.coolwarm)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=-60)
plt.yticks(tick_marks, classes)
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.tight_layout()

# thresh = cm.max() / 2.
# for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#     plt.text(j, i, cm[i, j],
#              horizontalalignment='center',
#              verticalalignment='center',
#              color='white' if cm[i, j] > thresh else 'black')

plt.savefig('confusion_matrix.png')
plt.show()

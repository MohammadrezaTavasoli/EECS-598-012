import numpy as np
import cPickle as pkl

np.random.seed(seed=1234)

x = np.random.random((1000, 20))

w1 = np.random.random((20, 10))
b1 = np.random.random((10,))
w2 = np.random.random((10,))
b2 = np.random.random((1,))

y = np.matmul(x, w1) + b1
z = np.maximum(y, 0)
o = np.matmul(z, w2) + b2

o = o - np.min(o)
o = o / np.max(o)

labels = o > 0.5
labels = labels.astype('int')

print(np.sum(labels))

x = x + np.random.random((1000, 20))/8

pkl.dump((x, labels), open('data.pkl', 'w'))

# 3.6.1 MNIST data set

'''
Download from
https://github.com/oreilly-japan/deep-learning-from-scratch
'''

import os
import sys

sys.path.append(os.pardir)
from dataset.mnist import load_mnist

# takes a few minuets
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

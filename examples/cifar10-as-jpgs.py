"""Save subset of cifar10 dataset as jpgs in labeled folders."""

import numpy as np
import os
from keras.datasets import cifar10
from scipy import misc


n = 100 # number of images to save per class

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 
               'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# create folder structure
for name in class_names:
    path = os.path.join('cifar10-subset', name)
    if not os.path.exists(path):
        os.makedirs(path)

# randomly select subset of images
idxs = []
for i in range(len(class_names)):
    idxs += np.random.choice(
            np.where(y_train == i)[0], 
            n, replace=False).tolist()

# save images as jpgs
for i in range(len(idxs)):
    class_name = class_names[int(y_train[idxs[i]])]
    path = os.path.join('cifar10-subset', class_name, str(i) + '.jpg')
    misc.imsave(path, x_train[idxs[i]])


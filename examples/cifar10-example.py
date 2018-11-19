"""Cross validate transfer learning on subset of cifar10 dataset.

Uses 100 random images from each cifar10 class. Shows how to:
    - extract features from a pre-trained model to train a new model
    - setup and use cross_validation function on new model
    - combine classes
    - exclude classes
"""

import os
import shutil
import subprocess
import sys

import keras
from keras import backend as K
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, Input
from keras.models import Model, Sequential

# import modules from parent directory
sys.path.append('../')
import report
import retrain


# download and save a subset of cifar10 if needed
if not os.path.exists('cifar10-subset'):
    print("Saving subset of cifar10 images as jpgs...\n")
    subprocess.call('python ./cifar10-as-jpgs.py', shell=True)

# setup paths
data_dir = 'cifar10-subset' # contains images in labeled folders
tmp_dir = 'tmp'
bottleneck_file = os.path.join(tmp_dir, 'cifar10-subset-vgg16-bottlenecks.h5')

# remove any results files from previous runs
results_dir = os.path.join(tmp_dir, 'results')
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)

# save terminal output to file
sys.stdout = report.Logger(results_dir)

# display summary of images in data_dir
report.data_summary(data_dir)

# select base model for feature extraction,
# we'll use a VGG16 model that has been pretrained on the ImageNet dataset
base_model = retrain.load_base_model('VGG16', input_shape=(64, 64, 3))

# save data to bottleneck file if file doesn't already exist (caches features)
retrain.create_bottlenecks(bottleneck_file, data_dir, base_model)

# define the classes to combine into 2 new classes
combine = {'animals': ('cat', 'deer', 'dog', 'frog', 'horse'),
           'not-animals': ('airplane', 'automobile','ship', 'truck')}

# define classes to exclude
exclude = ('bird', )

# define Keras model
# the model will be trained on features stored in the bottleneck file
# and will classify images into the 2 new classes we defined
model = Sequential(name='final_layers')
model.add(Dense(
    units=256,
    activation='relu',
    input_shape=base_model.output_shape[1:]))
model.add(Dropout(0.5))
# want to classify the 2 new classes
model.add(Dense(units=2, activation='softmax'))

# same model using Functional API
#inputs = Input(shape=base_model.output_shape[1:])
#x = Dense(256, activation='relu')(inputs)
#x = Dropout(0.5)(x)
#predictions = Dense(2, activation='softmax')(x)
#model = Model(inputs=inputs, outputs=predictions, name='final_layers')

# define optimizer
optimizer = optimizers.Adam()

# cross validate model using oversampling to balance classes,
# see the docstring of retrain.cross_validate() for more
# cross validation options
retrain.cross_validate(
        model, optimizer, bottleneck_file, tmp_dir, data_dir,
        combine=combine, exclude=exclude, resample=1.0,
        base_model=base_model, summarize_model=True,
        summarize_misclassified_images=True)

K.clear_session() # prevent TensorFlow error message


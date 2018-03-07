# Cross Validate Transfer Learning
Uses the cross validation functions from scikit-learn to evaluate image classification transfer learning with Keras models.

The system has the following features:
- transfer learning bottlenecks stored in the HDF5 format for increased speed using h5py
- stratified k-fold, group k-fold, and leave one group out cross validation
- groups can be defined in csvs so that cross validation splits are done on group membership
- combine or exclude classes (so you don't have to copy or delete images on disk)
- ways to deal with unbalanced classes
    - oversampling to balance classes during training
    - use class balance to scale the loss function during training
- logger to save all terminal output to a log file

Results (particularly individual class scores for each fold) are saved as csvs for further analysis in your platform of choice.
Training status, accuracy, F-scores, and confusion matrices are output to terminal while the tests are running.
Additional info, such as model or data summaries, can also be output to terminal.

### Setup
Install [miniconda](http://conda.pydata.org/miniconda.html).

Create a conda environment:

    conda create -n Keras python=3 numpy scipy yaml h5py scikit-learn pillow matplotlib tensorflow keras
    source activate Keras 

### Usage
Activate the conda environment:

    source activate Keras

See cifar10-example.py for a usage example. Also most functions have up to date docstrings on the master branch.

##### Preparing Images
Put the images in folders named with the image class label. Then put these folders in a parent directory so the directory structure looks something like:

    images
    ├── airplane
    ├── automobile
    ├── bird
    ├── cat
    ├── deer
    ├── dog
    ├── frog
    ├── horse
    ├── ship
    └── truck


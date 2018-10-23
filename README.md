# Cross Validate Transfer Learning
Use the cross validation functions from scikit-learn to evaluate image classification transfer learning with Keras models.

The system has the following features:
- Transfer learning bottlenecks stored in the HDF5 format for increased speed using h5py.
- Stratified k-fold, group k-fold, and leave one group out cross validation.
- Groups can be defined in csvs so that cross validation splits are done on group membership.
- Combine or exclude classes (so you don't have to copy or delete images on disk).
- Ways to deal with unbalanced classes:
    - Oversampling to balance classes during training.
    - Use class balance to scale the loss function during training.
- Logger to save all terminal output to a log file.

Results (particularly individual class scores for each fold) are saved as csvs for further analysis in your platform of choice.
Training status, accuracy, F-scores, and confusion matrices are output to terminal while the tests are running.
Additional info, such as model or data summaries, can also be output to terminal.

## Quick Intro
[CIFAR-10 Example Notebook](./examples/cifar10-example.ipynb)

## Setup
Install [miniconda](http://conda.pydata.org/miniconda.html).

Create a conda environment:

    conda create -n Keras python=3 numpy scipy yaml h5py scikit-learn pillow matplotlib tensorflow keras
    source activate Keras 

## Usage
See the example notebook and script. Most functions have detailed docstrings.

### Preparing Images
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

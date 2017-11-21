# Cross Validation of Transfer Learning with Keras & Scikit-learn
Use the cross validation functions from scikit-learn to evaluate 
convolutional neural network models from Keras. 

The system has the following features:
- transfer learning bottlenecks stored in the HDF5 format for increased speed using h5py
- define groups so that cross validation splits are done on group membership
- combine or exclude classes (so you don't have to copy or delete images on disk)
- oversampling to balance classes
- other stuff I might be forgetting

Results (particularly individual class scores for each fold) are saved as csvs for further analysis in your platform of choice.
F-scores and confusion matrices are output to the console while the tests are running.

### Setup
Install [miniconda](http://conda.pydata.org/miniconda.html).

Create a conda environment:

    conda create -n Keras python=3.5 numpy scipy yaml h5py scikit-learn pillow
    source activate Keras 
    pip install tensorflow
    pip install keras

*Note some of the required conda packages might be missing from this list!*

### Usage
Activate the conda environment:

    source activate Keras

See ssu_preserves.py for a usage example. Also most functions have up to date docstrings on the master branch.

##### Preparing Images
Put the images in folders named with the image class label. 
Each class needs at least 25 images.
Then put these folders in a parent directory so the directory structure looks something like:

    images
    ├── bobcat
    ├── deer
    ├── human
    ├── nothing
    ├── possum
    ├── skunk
    ├── squirrel
    └── turkey


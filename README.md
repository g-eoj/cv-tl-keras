# Cross Validation of Transfer Learning with Keras
Generate F-Scores, Accuracy, and Confusion Matrices to help evaluate CNN models.

### Setup (missing some conda packages)
Install [miniconda](http://conda.pydata.org/miniconda.html).

Create a conda environment:

    conda create -n Keras python=3.5 numpy scipy yaml h5py scikit-learn pillow
    source activate Keras 
    pip install tensorflow
    pip install keras

### Usage
Activate the conda environment:

    source activate Keras

Most functions should have up to date docstrings on the master branch.
See ssu_preserves.py for usage example.

#### Preparing Images
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


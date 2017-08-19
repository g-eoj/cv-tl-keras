from keras import backend as K
from keras import optimizers
from retrain import create_bottlenecks, cross_validate, \
        load_base_model, train_and_evaluate, group_dict
import report

from keras.models import Model, Sequential
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input

import numpy as np
import os.path

def create_groups(data_dir, groups_file):
    """Save csv (file_name,patient) of patient grouping."""

    print("\nCreating patient groups...")
    if not os.path.exists(groups_file):
        # load images
        from keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator()
        images = datagen.flow_from_directory(
            data_dir,
            batch_size=1,
            class_mode='categorical',
            shuffle=False)

        file_names = images.filenames

        group = []    
        for name in file_names:
            group.append(name.split('patient')[1][0:-4])

        groups = np.hstack((np.array(file_names).reshape((-1,1)), np.array(group).reshape((-1,1))))
        np.savetxt(groups_file, groups, delimiter=',', fmt='%s')
        print("Done.")
    else:
        print("Patient groups already exist.")


# load base model
#input_shape = (160, 160, 3)
input_shape = None
base_model = load_base_model('InceptionV3', input_shape)
#base_model = load_base_model('ResNet50', input_shape)
#base_model = load_base_model('VGG16')

# extract features from an earlier InceptionV3 layer
base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed7').output, name='inception_v3')
print(base_model.output.name, "layer will be used for creating bottlenecks.")  
x = base_model.output
x = GlobalAveragePooling2D()(x)
base_model = Model(inputs=base_model.input, outputs=x, name='inception_v3')
#base_model.summary()

# setup paths
#data_dir = './research/ssu_ct-scans/paper-rescale-full'
#data_dir = './research/ssu_ct-scans/paper-rescale'
#data_dir = './research/ssu_ct-scans/gs_bicubic_-1400to240'
#data_dir = './research/ssu_ct-scans/gs_-1400to240'
#data_dir = './research/ssu_ct-scans/even-split_-1400to200'
data_dir = './research/ssu_ct-scans/even-split_-1000to200'
tmp_dir = './research/ssu_ct-scans/tmp/'
log_dir = tmp_dir + 'logs/'
groups_file = './research/ssu_ct-scans/patient-groups.csv' # csv -> file_name,group
#bottleneck_file = tmp_dir + base_model.name + '-mixed7-bottlenecks-gs_-1400to240.csv'
bottleneck_file = './research/ssu_ct-scans/tmp/inception_v3-mixed7-bottlenecks-even-split_-1000to200.h5'

# create groups files
create_groups(data_dir, groups_file)
print()

report.data_summary(data_dir, groups_file, csv=tmp_dir+'data_summary.csv')

# get/create bottlenecks 
groups_files = [groups_file]
bottlenecks = create_bottlenecks(bottleneck_file, data_dir, base_model, groups_files)

# perform tests
cv = True
groups = "patient-groups"
optimizer = optimizers.Adam(clipnorm=1.0)
if not cv:
    train_and_evaluate(
            base_model, bottlenecks, tmp_dir, log_dir, 
            test_size=0.3, groups=groups, use_weights=True,
            optimizer=optimizer, dropout_rate=0.5, epochs=20, batch_size=512,
            save_model=False)
else:
    cross_validate(
            base_model, bottlenecks, groups=groups, 
            num_folds=5, logo=False, use_weights=False, resample=1.0,
            optimizer=optimizer, dropout_rate=0.5, epochs=20, batch_size=512,
            summarize_model=True, summarize_misclassified_images=False)

K.clear_session() # prevent TensorFlow error message


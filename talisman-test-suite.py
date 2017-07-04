from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from retrain import create_bottlenecks, k_fold_cross_validate, \
        load_base_model, train_and_evaluate_final_layers

import numpy as np
import os.path

def create_groups(data_dir, groups_file):
    """Save csv (file_name,patient) of patient grouping."""

    print("Creating patient groups...")
    if not os.path.exists(groups_file):
        # load images
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


def group_dict(groups_file):
    """Returns dictionary of group membership."""

    _ = np.loadtxt(groups_file, delimiter=',', dtype='U')
    groups = {}
    for file_name, group in _:
        groups[file_name] = group

    return groups

# load base model
base_model = load_base_model('InceptionV3')
#base_model = load_base_model('ResNet50')
#base_model = load_base_model('VGG16')

# setup paths
data_dir = './research/ssu_ct-scans/paper-rescale-full'
tmp_dir = './research/ssu_ct-scans/tmp/'
log_dir = tmp_dir + 'logs/'
bottleneck_file = tmp_dir + base_model.name + '-bottlenecks-paper-rescale-full.csv'
class_file = tmp_dir + base_model.name + '-retrained-classes-paper-rescale-full.csv'
groups_file = './research/ssu_ct-scans/paper-rescale-full-patient-groups.csv' # csv -> file_name,group

# get bottlenecks and classes
create_groups(data_dir, groups_file)
groups = group_dict(groups_file)
bottlenecks, classes = create_bottlenecks(
        bottleneck_file, class_file, data_dir, base_model, groups)

# perform tests
cross_validate = True
if not cross_validate:
    train_and_evaluate_final_layers(
            base_model, bottlenecks, tmp_dir, log_dir, 
            learning_rate=0.001, dropout_rate=0.7, epochs=20, batch_size=20, test_size=0.2,
            save_model=False)
else:
    k_fold_cross_validate(
            base_model, bottlenecks, classes, num_folds=5, 
            learning_rate=0.001, dropout_rate=0.7, epochs=20, batch_size=20,
            summarize_model=True, summarize_misclassified_images=False)

K.clear_session() # prevent TensorFlow error message


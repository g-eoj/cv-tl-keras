import numpy as np
from keras import backend as K
from keras import optimizers
from retrain import create_bottlenecks, cross_validate, \
        load_base_model, train_and_evaluate, group_dict


# load base model
base_model = load_base_model('InceptionV3')
#base_model = load_base_model('ResNet50')
#base_model = load_base_model('VGG16')

# setup paths
data_dir = './research/ssu_preserves/training_images'
tmp_dir = './research/ssu_preserves/tmp/'
log_dir = tmp_dir + 'logs/'
bottleneck_file = tmp_dir + base_model.name + '-groups-bottlenecks.txt'
class_file = tmp_dir + base_model.name + '-retrained-classes.txt'
groups_file = './research/ssu_preserves/image_groups.txt' # csv -> file_name,group

# get bottlenecks and classes
groups = group_dict(groups_file)
bottlenecks, classes = create_bottlenecks(
        bottleneck_file, class_file, data_dir, base_model, groups)

# perform tests
cv = True
#optimizer = 'adam'
optimizer = optimizers.Adam(clipnorm=1.0)
if not cv:
    train_and_evaluate(
            base_model, bottlenecks, tmp_dir, log_dir, 
            test_size=0.2, use_groups=True, use_weights=False,
            optimizer=optimizer, dropout_rate=0.8, epochs=20, batch_size=64,
            save_model=False)
else:
    cross_validate(
            base_model, bottlenecks, classes, 
            num_folds=5, logo=False, use_weights=True, resample=0.5,
            optimizer=optimizer, dropout_rate=0.8, epochs=20, batch_size=64,
            summarize_model=True, summarize_misclassified_images=False)

K.clear_session() # prevent TensorFlow error message


import exifread as EXIF
import os.path
import numpy as np
import report
from datetime import datetime
from keras import backend as K
from keras import optimizers
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
from retrain import create_bottlenecks, cross_validate, \
        load_base_model, train_and_evaluate, group_dict, \
        Logger

def create_camera_groups(data_dir, groups_file):
    """Save csv (file_name,camera_name) of camera grouping."""

    print("\nCreating camera groups...")
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
            group.append(name.split('_')[0].split('/')[1])

        groups = np.hstack((np.array(file_names).reshape((-1,1)), np.array(group).reshape((-1,1))))
        np.savetxt(groups_file, groups, delimiter=',', fmt='%s')
        print("Done.")
    else:
        print("Camera groups already exist.")


def create_capture_event_groups(data_dir, groups_file, times_file, time_stamp_difference=2):
    """Save csv (file_name,capture_event_index) of capture event grouping."""

    print("\nCreating capture event groups...")
    if not os.path.exists(groups_file):
        # load images
        from keras.preprocessing.image import ImageDataGenerator
        datagen = ImageDataGenerator()
        images = datagen.flow_from_directory(
            data_dir,
            batch_size=1,
            class_mode='categorical',
            shuffle=False)

        # detecting capture events requires sorted file names
        file_names = sorted(images.filenames)

        group = []    
        initial_image = True
        capture_event_index = 0
        time_taken = []
        for name in file_names:
            path = os.path.join(data_dir, name)
            with open(path , 'rb') as fh:
                tags = EXIF.process_file(fh, stop_tag="EXIF DateTimeOriginal")
                date_taken = tags["EXIF DateTimeOriginal"]
                time_taken.append(date_taken)
            new = datetime.strptime(str(date_taken), "%Y:%m:%d %H:%M:%S")
            if initial_image:
                group.append(capture_event_index)
                initial_image = False
            else:
                difference = abs(new-old)
                if difference.seconds <= time_stamp_difference:
                    group.append(capture_event_index)
                else:
                    capture_event_index += 1
                    group.append(capture_event_index)
            old = datetime.strptime(str(date_taken), "%Y:%m:%d %H:%M:%S")

        groups = np.hstack((np.array(file_names).reshape((-1,1)), np.array(group).reshape((-1,1))))
        times = np.hstack((np.array(file_names).reshape((-1,1)), np.array(time_taken).reshape((-1,1))))
        np.savetxt(groups_file, groups, delimiter=',', fmt='%s')
        np.savetxt(times_file, times, delimiter=',', fmt='%s')
        print("Done.")
    else:
        print("Capture event groups already exist.")


# setup paths
data_dir = './research/ssu_preserves/images'
tmp_dir = './research/ssu_preserves/tmp/'
log_dir = tmp_dir + 'logs/'
camera_groups_file = './research/ssu_preserves/camera_groups.csv' # csv: file_name,group
capture_event_groups_file = './research/ssu_preserves/capture_event_groups.csv' # csv: file_name,group
times_file = './research/ssu_preserves/times_taken.csv'

# setup logging / flush previous results
import shutil
import sys
if os.path.exists(tmp_dir + 'results'):
    shutil.rmtree(tmp_dir + 'results')
os.makedirs(tmp_dir + 'results')
sys.stdout = Logger(tmp_dir + 'results/')

# load base model
base_model = load_base_model('InceptionV3')
#base_model = load_base_model('ResNet50')
#base_model = load_base_model('VGG16')

# extract features from an earlier InceptionV3 layer
#base_model = Model(inputs=base_model.input, outputs=base_model.get_layer('mixed7').output, name='inception_v3')
#print(base_model.output.name, "layer will be used for creating bottlenecks.")  
#x = base_model.output
#x = GlobalAveragePooling2D()(x)
#base_model = Model(inputs=base_model.input, outputs=x, name='inception_v3')

# setup bottleneck path
bottleneck_file = tmp_dir + base_model.name + '-' + base_model.layers[-1].name + '-bottlenecks.h5'

# create groups files
create_camera_groups(data_dir, camera_groups_file)
create_capture_event_groups(data_dir, capture_event_groups_file, times_file)
print()

report.data_summary(data_dir, camera_groups_file, csv=tmp_dir+'data_summary.csv')

# get/create bottlenecks 
groups_files = [camera_groups_file, capture_event_groups_file]
bottlenecks = create_bottlenecks(bottleneck_file, data_dir, base_model, groups_files)

# perform tests
cv = True
#groups = "camera_groups"
groups = "capture_event_groups"

# Binary
combine = {'something': ('deer', 'squirrel', 'human', 'rabbit', #'unknown', 
        'turkey', 'skunk', 'bobcat', 'possum', 'coyote', 'fox', 'dog', 'raven', 
        'miscellaneous birds', 'quail', 'multiple', 'mountainlion', 'vehicle', 
        'mouse', 'raccoon', "stellar's jay", 'crow', 'hawk', 'owl', 'pig')}
exclude = ('unknown',)

# Binary with exclusion
#combine = {'something': ('deer', 'squirrel', 'human', 'rabbit', #'unknown', 
#        'turkey', 'skunk', 'bobcat', 'possum', 'coyote')} 
#exclude = ('unknown',
#        'fox', 'dog', 'raven', 
#        'miscellaneous birds', 'quail', 'multiple', 'mountainlion', 'vehicle', 
#        'mouse', 'raccoon', "stellar's jay", 'crow', 'hawk', 'owl', 'pig')

# Mixed
#combine = {'mixed': ('raven', 'miscellaneous birds', 'quail', "stellar's jay", 'crow', 'hawk', 'owl', 
#    'dog', 'multiple', 'mountainlion', 'vehicle', 'mouse', 'raccoon', 'pig', # < 75 
#    'fox', # < 100
#    'coyote', 'possum', 'bobcat', 'skunk')} # < 600 
#exclude = ('unknown',)

# Exclusion
#combine = {'birds': ('raven', 'miscellaneous birds', 'quail', "stellar's jay", 'crow', 'hawk', 'owl')}
#exclude = ('dog', 'multiple', 'mountainlion', 'vehicle', 'mouse', 'raccoon', 'pig', # < 75 
#    'unknown')

optimizer = None
if not cv:
    train_and_evaluate(
            base_model, bottlenecks, tmp_dir, log_dir, combine=combine, exclude=exclude,
            test_size=0.2, groups=groups, use_weights=False, resample=1.0,
            optimizer=optimizer, dropout_rate=0.7, epochs=20, batch_size=512,
            save_model=False)
else:
    cross_validate(
            base_model, bottlenecks, tmp_dir, data_dir,
            groups=groups, combine=combine, exclude=exclude,
            num_folds=5, logo=False, use_weights=False, resample=1.0,
            optimizer=optimizer, dropout_rate=0.5, epochs=3, batch_size=64,
            summarize_model=True, summarize_misclassified_images=True)

K.clear_session() # prevent TensorFlow error message


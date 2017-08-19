import exifread as EXIF
import os.path
import numpy as np
import report
from datetime import datetime
from keras import backend as K
from keras import optimizers
from retrain import create_bottlenecks, cross_validate, \
        load_base_model, train_and_evaluate, group_dict

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


def create_capture_event_groups(data_dir, groups_file, time_stamp_difference=2):
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
        for name in file_names:
            path = os.path.join(data_dir, name)
            with open(path , 'rb') as fh:
                tags = EXIF.process_file(fh, stop_tag="EXIF DateTimeOriginal")
                date_taken = tags["EXIF DateTimeOriginal"]
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
        np.savetxt(groups_file, groups, delimiter=',', fmt='%s')
        print("Done.")
    else:
        print("Capture event groups already exist.")


# load base model
base_model = load_base_model('InceptionV3')
#base_model = load_base_model('ResNet50')
#base_model = load_base_model('VGG16')

# setup paths
data_dir = './research/ssu_preserves/images'
tmp_dir = './research/ssu_preserves/tmp/'
log_dir = tmp_dir + 'logs/'
camera_groups_file = './research/ssu_preserves/full_camera_groups.csv' # csv: file_name,group
capture_event_groups_file = './research/ssu_preserves/full_capture_event_groups.csv' # csv: file_name,group
bottleneck_file = tmp_dir + base_model.name + '-' + base_model.layers[-1].name + '-full-bottlenecks.h5'

# create groups files
create_camera_groups(data_dir, camera_groups_file)
create_capture_event_groups(data_dir, capture_event_groups_file)
print()

report.data_summary(data_dir, camera_groups_file, csv=tmp_dir+'data_summary.csv')

# get/create bottlenecks 
groups_files = [camera_groups_file, capture_event_groups_file]
bottlenecks = create_bottlenecks(bottleneck_file, data_dir, base_model, groups_files)

# perform tests
cv = True
groups = "capture_event_groups"
optimizer = optimizers.Adam(clipnorm=1.0)
if not cv:
    train_and_evaluate(
            base_model, bottlenecks, tmp_dir, log_dir, 
            test_size=0.2, groups=groups, use_weights=True,
            optimizer=optimizer, dropout_rate=0.8, epochs=20, batch_size=512,
            save_model=False)
else:
    cross_validate(
            base_model, bottlenecks, groups=groups,
            num_folds=5, logo=False, use_weights=False, resample=1.0,
            optimizer=optimizer, dropout_rate=0.8, epochs=20, batch_size=512,
            summarize_model=True, summarize_misclassified_images=True)

K.clear_session() # prevent TensorFlow error message


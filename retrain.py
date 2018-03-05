import h5py
import numpy as np
import os
import os.path

import keras
from keras import backend as K
from keras import optimizers
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import preprocess_input as imagenet_utils_preprocess_input
from keras.applications.inception_v3 import preprocess_input as inception_v3_preprocess_input
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from report import save_model_summary, print_confusion_matrix, print_model_info, print_class_balance

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit, GroupKFold, LeaveOneGroupOut
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.utils import class_weight as cw


os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Suppress TensorFlow message about CPU features


def load_base_model(model_name, input_shape=None):
    """Load pre-trained model without final layers."""

    # Need include_top=False and pooling='avg' to generate bottleneck features
    if model_name == 'InceptionV3':
        if input_shape is None:
            input_shape = (299, 299, 3)
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    elif model_name == 'ResNet50':
        if input_shape is None:
            input_shape = (224, 224, 3)
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    elif model_name == 'VGG16':
        if input_shape is None:
            input_shape = (224, 224, 3)
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape, pooling='avg')
    else:
        print("Model name not recognized.")
        return
    print('\n' + base_model.name, 'base model with input shape', base_model.input_shape, 'loaded.')
    return base_model


def group_dict(groups_file):
    """Returns dictionary of group membership, where the keys are file names.

    input: file path to csv which have rows with the format: file_name,group
    """

    print("Loading groups...")
    _ = np.loadtxt(groups_file, delimiter=',', dtype='U')
    groups = {}
    for file_name, group in _:
        groups[file_name] = group
    print(groups_file, "loaded.")

    return groups


def create_bottlenecks(bottleneck_file, data_dir, base_model, groups_files=[]):
    """Saves features and related data to 'bottleneck_file'.

    Generates features for the images in 'data_dir' with 'base_model' and saves
    the features along with related data to 'bottleneck_file' using the HDF5 
    data format. 'bottleneck_file' can be loaded as an h5py file object which 
    works like a dictionary. For example if 'bottleneck_file' is loaded into 
    the variable bottlenecks, then bottlenecks['features'][:] returns a numpy 
    array of the features. The keys available are:
        'base_model' -> base_model.name
        'features_layer' -> base_model.layers[-1].name
        'file_names' -> np.array(file_names, dtype='S')
        'class_numbers' -> class_numbers (numpy array)
        'class_labels' -> np.array(class_labels, dtype='S')
        'classes' -> np.array(classes, dtype='S')
        'features' -> features (numpy array)
        'blank_groups' -> numpy array, used for sklearn's cross validation
        Each group type also gets a key, for example the groups file
            patient_groups.csv
        will cause creation of the key
            'patient_groups'

    Inputs:
        bottleneck_file: path to h5 file to be created
        data_dir: path to directory of images used to calculate features 
            (where images are in a subdirectory for each class)
        base_model: Keras model used to generate features
        groups (optional): list of file paths to csvs which 
            have rows with the format: file_name,group

    """

    print("Generating bottleneck file... ")
    if not os.path.exists(bottleneck_file):
        # Use correct image preprocessing for model
        if base_model.name in ('inception_v3'):
            preprocess_input = inception_v3_preprocess_input
        elif base_model.name in ('vgg16', 'vgg19', 'resnet50'):
            preprocess_input = imagenet_utils_preprocess_input
        else:
            print(base_model.name, "preprocessing function not found. Exiting.")
            return
        
        img_height, img_width = base_model.input_shape[1], base_model.input_shape[2]
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        images = datagen.flow_from_directory(
            data_dir,
            target_size=(img_height, img_width),
            batch_size=1,
            class_mode='categorical',
            shuffle=False)

        # Keras orders classes alphanumerically
        classes = sorted(images.class_indices.keys()) 

        file_names = images.filenames
        class_numbers = images.classes
        class_labels = []
        for number in class_numbers:
            class_labels.append(classes[number])
        features = base_model.predict_generator(images, images.samples, verbose=1)

        bottlenecks = h5py.File(bottleneck_file, 'w')
        bottlenecks.attrs['base_model'] = base_model.name
        bottlenecks.attrs['features_layer'] = base_model.layers[-1].name
        bottlenecks.create_dataset('file_names', data=np.array(file_names, dtype='S'))
        bottlenecks.create_dataset('class_numbers', data=class_numbers)
        bottlenecks.create_dataset('class_labels', data=np.array(class_labels, dtype='S'))
        bottlenecks.create_dataset('classes', data=np.array(classes, dtype='S'))
        bottlenecks.create_dataset('features', data=features)

        # blank group labels sometimes required for the way sklearn's cross validation is called
        group_labels = ['' for name in file_names]
        bottlenecks.create_dataset('blank_groups', data=np.array(group_labels, dtype='S'))
        for groups_file in groups_files:
            groups_type = groups_file.split('/')[-1].split('.')[0]
            groups = group_dict(groups_file)
            group_labels = [groups[name] for name in file_names] 
            bottlenecks.create_dataset(groups_type, data=np.array(group_labels, dtype='S'))

        bottlenecks.close()
        print(bottleneck_file, "created.\n")
    else:
        print(bottleneck_file, "already exists.\n")


def load_bottlenecks(bottleneck_file):
    """Loads 'bottleneck_file' into an h5py file object and returns it.

    Inputs:
        bottleneck_file: path to h5 file 

    Returns: h5py file object
    """

    print("Loading ", bottleneck_file, "...\n", sep='')
    return h5py.File(bottleneck_file, 'r')


def combine_classes(combine, bottlenecks):
    """Given a bottlenecks file, combine multiple classes into a single class.
    It's possible for multiple new combinations can be created at once.

    Inputs:
        combine: dictionary, values are existing class names that are to be
            combined into a new class with the name of their key.
        bottlenecks: h5py file object returned by 'create_bottlenecks' function

    Returns: tuple of numpy arrays, (class_numbers, class_labels, classes) which
        are meant to be used instead of the corresponding arrays from the 
        bottleneck file
    """

    class_labels = bottlenecks["class_labels"][:].astype(object)
    classes = bottlenecks["classes"][:].astype(object)
    # use 'object' dtype (i.e. strings are bytes objects) so string length 
    # can change in numpy arrays
    class_indices = {}
    for i, name in enumerate(classes):
        class_indices[name] = i

    for new_class_name in combine:
        print("\nMaking", new_class_name, "class from", combine[new_class_name])
        # use encode() so all strings are bytes objects
        combine_labels = sorted([name.encode() for name in combine[new_class_name]])
        combine_numbers = sorted([class_indices[name] for name in combine_labels])

        for number in combine_numbers:
            class_labels[class_labels == classes[number]] = new_class_name.encode()   

        # replace class name corresponding to smaller class number in classes
        classes[combine_numbers[0]] = new_class_name.encode()
        # delete class names corresponding to larger class numbers from classes
        classes = np.sort(np.delete(classes, combine_numbers[1:]))
        # update class indices
        class_indices = {}
        for i, name in enumerate(classes):
            class_indices[name] = i

    print("Updating class numbers...")
    class_numbers = np.array([class_indices[name] for name in class_labels])

    # convert bytes objects back to fixed length strings for compatability and speed
    return class_numbers, class_labels.astype(str), classes.astype(str) 


def exclude_classes(exclude, class_labels):
    """Returns indexes corresponding to classes that are to be excluded.
    
    Inputs:
        exclude: tuple of strings, class names to be excluded 
        class_labels: 'class_labels' numpy array from 
            'bottlenecks' h5py object

    Returns: numpy array of indexes
    """

    excluded = []
    for name in exclude:
        indexes = np.where(class_labels == name)[0]
        excluded = np.concatenate((excluded, indexes))
    return excluded.astype(int)


def exclude_groups(exclude, group_labels):
    """Returns indexes corresponding to classes that are to be excluded.
    
    Inputs:
        exclude: tuple of strings, class names to be excluded 
        group_labels: 'group_labels' numpy array from 
            'bottlenecks' h5py object

    Returns: numpy array of indexes
    """

    excluded = []
    for name in exclude:
        indexes = np.where(group_labels == name)[0]
        excluded = np.concatenate((excluded, indexes))
    return excluded.astype(int)


def create_final_layers(base_model, num_classes, 
        optimizer=None, learning_rate=0.001, dropout_rate=0.5):
    """Returns a Keras model that is meant to be trained with features 
    from 'base_model'.
    
    Inputs:
        base_model: Keras model used to generate features, it's assumed the 
            model's output is a vector
        num_classes: int, count of the number of classes in training data
        optimizer (optional): Keras optimizer
        learning_rate (optional)
        dropout_rate (optional)

    Returns: trainable Keras model 
    """

    # setup final layers using sequential model
    model = Sequential(name='final_layers')
    model.add(Dense(
        base_model.output_shape[1] // 2, 
        activation='relu', 
        input_shape=base_model.output_shape[1:]))
    model.add(Dropout(dropout_rate))
    model.add(Dense(num_classes, activation='softmax'))

    # setup final layers using funcional API and return whole model
    #inputs = Input(shape=base_model.output_shape[1:])
    #x = Dense(base_model.output_shape[1] // 2, activation='relu')(inputs)
    #x = Dropout(dropout_rate)(x)
    #predictions = Dense(num_classes, activation='softmax')(x)
    #model = Model(inputs=inputs, outputs=predictions, name='final_layers')
        
    # compile the final layers model
    if optimizer is None:
        optimizer = keras.optimizers.Adam(clipnorm=1.0)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def train_and_evaluate(
        base_model, bottlenecks, tmp_dir, log_dir, combine=None, exclude=None,
        test_size=0.1, groups=None, use_weights=False, resample=None,
        optimizer=None, learning_rate=0.001, dropout_rate=0.5, epochs=10, 
        batch_size=32, save_model=False):
    """Use a train-test split to evaluate final layers in transfer learning. 
    
    Prints training and results summary. If group labels exist, the train-test 
    split will be on groups.

    Inputs:
        base_model: Keras model used to generate features, it's assumed the 
            model's output is a vector
        bottlenecks: h5py file object returned by 'create_bottlenecks' function
        tmp_dir: path, trained model is saved here when 'save_model' is True
        log_dir: path, tensorboard logs are saved here
        combine (optional): dictionary, values are existing class names that are to be
            combined into a new class with the name of their key.
        exclude (optional): tuple of strings, class names to be ignored 
        test_size (optional): proportion of data to be used for testing
        groups (optional): string, key used to get groups data from 
            'bottlenecks', for example, to use data from the groups file 
            'patient_groups.csv' the key should be 'patient_groups'
        use_weights (optional): use class balance to scale the loss function 
            during training
        resample: float, (if groups is not None) oversamples so that the number 
            of training samples in each class is equal to 
            (reasample * largest training class size)
        optimizer (optional): Keras optimizer to use when training final layers
        learning_rate (optional): model hyperparameter
        dropout_rate (optional): model hyperparameter
        epochs (optional): training parameter
        batch_size (optional): training parameter
        save_model (optional): if True, the 'base_model' and trained final 
            layers are combined and saved as a complete model in 'tmp_dir'
    """

    if combine is not None:
        class_numbers, class_labels, _ = combine_classes(combine, bottlenecks)
    else:
        class_numbers = bottlenecks["class_numbers"][:]
        class_labels = bottlenecks["class_labels"][:].astype(str)
    if groups is not None:
        group_labels = bottlenecks[groups][:].astype(str)
    else:
        group_labels = bottlenecks["blank_groups"][:].astype(str)
    features = bottlenecks["features"][:]
    bottlenecks.close()

    if exclude is not None:
        print("Removing", exclude, "classes.")
        excluded = exclude_classes(exclude, class_labels)
        class_labels = np.delete(class_labels, excluded, 0)
        group_labels = np.delete(group_labels, excluded, 0)
        features = np.delete(features, excluded, 0)
        classes = sorted(np.unique(class_labels))
        class_indices = {}
        for i, name in enumerate(classes):
            class_indices[name] = i
        print("Updating class numbers...")
        class_numbers = np.array([class_indices[name] for name in class_labels])

    num_classes = len(np.unique(class_numbers))
    
    # split bottlenecks into train and validation sets
    if group_labels[0] == '':
        train_features, validation_features, train_labels, validation_labels = \
                train_test_split(features, class_numbers, test_size=test_size)
    else:
        train, validate = next(GroupShuffleSplit(n_splits=2, test_size=test_size).split(
                features, class_numbers, group_labels))
        if resample is not None:
            uniques, counts = np.unique(class_numbers[train], return_counts=True)
            print("Oversampling to balance classes in training set.")
            print("Training set sizes will be at least ", 
                  resample, " times max training set class size of ", max(counts), ".", sep='')
            class_counts = dict(zip(uniques, counts))
            max_sample_size = int(max(counts) * resample)
            random_idxs = []
            for class_number in class_counts.keys():
                indexes = np.where(class_number == class_numbers)[0]
                indexes = np.intersect1d(indexes, train)
                if class_counts[class_number] < max_sample_size:
                    sample_size = max_sample_size - class_counts[class_number]
                    random_idxs = np.concatenate(
                        (random_idxs, np.random.choice(indexes, sample_size, replace=True)))
                    print("Class number", class_number, "training set size changed:", 
                          class_counts[class_number], "->", class_counts[class_number] + sample_size)
            train = np.concatenate((train, random_idxs)).astype(int)
        train_features, validation_features, train_labels, validation_labels = \
                features[train], features[validate], \
                class_numbers[train], class_numbers[validate]

    # do one hot encoding for labels
    train_labels_one_hot, validation_labels_one_hot = \
            to_categorical(train_labels), to_categorical(validation_labels)

    print()
    print_class_balance(class_labels, class_numbers, 
                        [train_labels, validation_labels], ["Train","Validate"])

    # create final layers model
    final_layers = create_final_layers(
            base_model, num_classes, optimizer=optimizer, 
            learning_rate=learning_rate, dropout_rate=dropout_rate)

    # callbacks
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, 
            batch_size=20, write_graph=False, write_images=False)
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
            factor=0.5, patience=4, verbose=1)
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, 
            patience=15, verbose=1, mode='auto')

    class_weight = None
    if use_weights:
        class_weight = cw.compute_class_weight(
                'balanced', range(num_classes), train_labels)
        class_weight = dict(zip(range(num_classes), class_weight))
        print("Class Weights:", class_weight)

    # train final layers
    final_layers.fit(train_features, train_labels_one_hot,
        batch_size=batch_size,
        #callbacks=[reduce_lr, early_stop, tensorboard],
        #callbacks=[reduce_lr, tensorboard],
        #callbacks=[reduce_lr],
        class_weight=class_weight,
        epochs=epochs,
        shuffle=True,
        validation_data=(validation_features, validation_labels_one_hot),
        verbose=1)

    predictions = final_layers.predict(validation_features)
    prediction_labels = np.argmax(predictions, axis=1)
    beta = 1.0
    precision_scores, recall_scores, fbeta_scores, _ = \
            precision_recall_fscore_support(
                validation_labels, 
                prediction_labels,
                beta=beta,
                average=None, 
                labels=range(num_classes))
    print('\nValidation Accuracy:', round(accuracy_score(validation_labels, prediction_labels), 4))
    print('Average Precision:', round(np.mean(precision_scores), 4))
    print('Average Recall:', round(np.mean(recall_scores), 4))
    print('Average F-beta(', beta, ') Score: ', round(np.mean(fbeta_scores), 4), sep="")
    print('F-beta(', beta, ') Scores: [', end="  ", sep="")
    for x in fbeta_scores:
        print(round(x, 2), end="  ")
    print("]\n")
    cm = confusion_matrix(
            validation_labels, 
            prediction_labels,
            labels=range(num_classes))
    print_confusion_matrix(cm, np.unique(class_labels), normalize=False)
    print()

    # training parameters/config summary
    print_model_info(batch_size, epochs, learning_rate, dropout_rate, final_layers)

    # wip - create and save complete retrained model, suitable for fine-tuning 
    if save_model:
        x = base_model.output
        x = Dense(base_model.output_shape[1] // 2, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for i, layer in enumerate(reversed(final_layers.layers), 1):
            pretrained_weights = layer.get_weights()
            model.layers[-i].set_weights(pretrained_weights)

        # slow learning rate for fine-tuning
        from keras.optimizers import SGD
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

        model.save(tmp_dir + base_model.name + '-retrained-model.h5')
        print("\nModel saved to:", tmp_dir + base_model.name + '-retrained-model.h5')
        # TODO save txt file of class:number for use in prediction
        # capture retrained model architecture with final layers
        save_model_summary(tmp_dir + base_model.name + '-retrained-model-summary.txt', model)


def group_k_fold(num_folds, features, class_numbers, group_labels):
    """Work in progress, returns indexes for random folds, has issues if 
    group sizes are imbalanced...
    """

    classes = np.unique(class_numbers)
    #np.random.shuffle(classes)
    groups, group_counts = np.unique(group_labels, return_counts=True)
    #i = np.argsort(group_counts)
    #groups = groups[i]
    np.random.shuffle(groups)

    folds = dict(zip(range(1, num_folds+1), [[] for f in range(num_folds)]))
    #fold_groups = dict(zip(range(1, num_folds+1), [[] for f in range(num_folds)]))

    for n in classes:
        for g in groups:
            indexes = np.argwhere((class_numbers==n) & (group_labels==g))
            indexes = indexes.flatten().tolist()
            if len(indexes) != 0:
                smallest = min(folds, key=lambda k: len(folds[k]))
                folds[smallest].extend(indexes)

    return folds    


def cross_validate(
        base_model, bottleneck_file, tmp_dir, data_dir,
        groups=None, combine=None, exclude=None, 
        num_folds=5, logo=False, use_weights=False, resample=None, optimizer=None, 
        learning_rate=0.001, dropout_rate=0.5, epochs=10, batch_size=16,
        summarize_model=True, summarize_misclassified_images=False):
    """Use cross validation to evaluate final layers in transfer learning. 
    
    Prints validation and results summary. If group labels exist, the folds 
    will split on groups.

    Inputs:
        base_model: Keras model used to generate features, it's assumed the 
            model's output is a vector
        bottleneck_file: path to h5 file created by 'create_bottlenecks' function
        tmp_dir: path, 'results' directory of results saved here
        data_dir: path to directory of images used to calculate features 
            (where images are in a subdirectory for each class)
        groups (optional): string, key used to get groups data from 
            'bottlenecks', for example, to use data from the groups file 
            'patient_groups.csv' the key should be 'patient_groups'
        combine (optional): dictionary, values are existing class names that are to be
            combined into a new class with the name of their key.
        exclude (optional): tuple of strings, class names to be ignored
        num_folds (optional): number of folds to use
        logo (optional): do leave one group out cross validation
        use_weights (optional): use class balance to scale the loss function 
            during training
        resample: float, oversamples so that the number of training samples in
            each class is equal to (reasample * largest training class size)
        optimizer (optional): Keras optimizer to use when training final layers
        learning_rate (optional): model hyperparameter
        dropout_rate (optional): model hyperparameter
        epochs (optional): training parameter
        batch_size (optional): training parameter
        summarize_model (optional): prints hyperparamter and model summary
        summarize_misclassified_images (optional): saves list of misclassified 
            image file names and random sample of misclassified images as a 
            jpg in the 'results' folder
    """

    bottlenecks = load_bottlenecks(bottleneck_file)

    if combine is not None:
        class_numbers, class_labels, classes = combine_classes(combine, bottlenecks)
    else:
        class_numbers = bottlenecks["class_numbers"][:]
        class_labels = bottlenecks["class_labels"][:].astype(str)
        classes = bottlenecks["classes"][:].astype(str)

    file_names = bottlenecks["file_names"][:].astype(str)
    if groups is not None:
        group_labels = bottlenecks[groups][:].astype(str)
        ## temp hack to run tests on a single group
        #camera_groups_labels = bottlenecks["camera_groups"][:].astype(str)
    else:
        group_labels = bottlenecks["blank_groups"][:].astype(str)
    features = bottlenecks["features"][:]
    bottlenecks.close()

    ## temp hack to run tests on a single group
    #exclude_cameras = ('UpperMostROWCamera', 'NorthernTowerMeadowCamera', 
    #    'UpperROWWoodChipFieldCamera', 'UpperTrailCamera', 'SODPlotCamera')
    #print("Removing", exclude_cameras, "groups.")
    #excluded = exclude_groups(exclude_cameras, camera_groups_labels)
    #class_labels = np.delete(class_labels, excluded, 0)
    #file_names = np.delete(file_names, excluded, 0)
    #group_labels = np.delete(group_labels, excluded, 0)
    #features = np.delete(features, excluded, 0)
    #class_numbers = np.delete(class_numbers, excluded, 0)

    if exclude is not None:
        print("Removing", exclude, "classes.")
        excluded = exclude_classes(exclude, class_labels)
        class_labels = np.delete(class_labels, excluded, 0)
        file_names = np.delete(file_names, excluded, 0)
        group_labels = np.delete(group_labels, excluded, 0)
        features = np.delete(features, excluded, 0)
        classes = sorted(np.unique(class_labels))
        class_indices = {}
        for i, name in enumerate(classes):
            class_indices[name] = i
        print("Updating class numbers...")
        class_numbers = np.array([class_indices[name] for name in class_labels])

    actual_classes = []
    predicted_classes =[]
    prediction_scores = []
    accuracy_scores = []
    group_labels_test = []
    file_names_test = []
    splits = []
    split_names = []
    split_metrics = {}
    num_classes = len(set(class_numbers))

    if group_labels[0] == '':
        print('\nPerforming stratified ', num_folds, '-fold cross validation...', sep='')
        cv = StratifiedKFold(n_splits=num_folds, shuffle=True)
    elif not logo:
        print('\nPerforming group ', num_folds, '-fold cross validation...', sep='')
        cv = GroupKFold(n_splits=num_folds)
    else:
        print('\nPerforming leave one group out cross validation...', sep='')
        num_groups = len(set(group_labels))
        cv = LeaveOneGroupOut()


    # ROC only works for binary classification
    #from scipy import interp
    import matplotlib
    matplotlib.use('Agg') 
    import matplotlib.pyplot as plt
    #from sklearn.metrics import roc_curve, auc
    #tprs = []
    #aucs = []
    #mean_fpr = np.linspace(0, 1, 100)

    # cv.split will ignore group_labels if cv is StratifiedKFold
    for i, split in enumerate(cv.split(features, class_numbers, group_labels)):
        train, test = split[0], split[1] 

        if not logo:
            split_name = i+1
            print("Fold ", i+1, "/", num_folds, sep='')
        else:
            split_name = group_labels[test][0]
            print("Group ", i+1, "/", num_groups, " | Group Name: '", split_name, "'", sep='')

        #print("Before resample:")
        #print("File count of test:", len(set(file_names[test])))
        #print("File count of train:", len(set(file_names[train])))
        #print("File count of intersection  of test and train:", len(set(file_names[test]) & set(file_names[train])))
        if resample is not None:
            uniques, counts = np.unique(class_numbers[train], return_counts=True)
            print("Oversampling to balance classes in training set.")
            print("Training set sizes will be at least ", 
                  resample, " times max training set class size of ", max(counts), ".", sep='')
            class_counts = dict(zip(uniques, counts))
            max_sample_size = int(max(counts) * resample)
            random_idxs = []
            for class_number in class_counts.keys():
                indexes = np.where(class_number == class_numbers)[0]
                indexes = np.intersect1d(indexes, train)
                if class_counts[class_number] < max_sample_size:
                    sample_size = max_sample_size - class_counts[class_number]
                    random_idxs = np.concatenate(
                        (random_idxs, np.random.choice(indexes, sample_size, replace=True)))
                    print("Class number", class_number, "training set size changed:", 
                          class_counts[class_number], "->", class_counts[class_number] + sample_size)
            train = np.concatenate((train, random_idxs)).astype(int)
            #print("File count of test:", len(set(file_names[test])))
            #print("File count of train:", len(set(file_names[train])))
            #print("File count of intersection  of test and train:", len(set(file_names[test]) & set(file_names[train])))

        actual_classes.extend(class_numbers[test])
        group_labels_test.extend(group_labels[test])
        file_names_test.extend(file_names[test])
        #print("First 10 test files:", sorted(file_names[test])[0:10])
        splits.append(class_numbers[test])
        if not logo:
            split_names.append("Fold " + str(i+1))
        else:
            split_names.append("Group " + split_name)

        class_weight = None # reset class weights
        if use_weights:
            class_weight = cw.compute_class_weight(
                    'balanced', np.unique(class_numbers), class_numbers[train])
            class_weight = dict(zip(np.unique(class_numbers), class_weight))
            print("Class Weights:", class_weight)

        model = None # reset the model
        model = create_final_layers(
                base_model, num_classes, optimizer=optimizer, 
                learning_rate=learning_rate, dropout_rate=dropout_rate)
        model.fit(features[train],
                  to_categorical(class_numbers[train]),
                  batch_size=batch_size,
                  class_weight=class_weight,
                  epochs=epochs,
                  shuffle=True,
                  verbose=2)

        predictions = model.predict(features[test])

        # save csv with all scores for data analysis
        csv = np.hstack((file_names[test].reshape((-1,1)),
                      class_labels[test].reshape((-2,1)),
                      group_labels[test].reshape((-1,1)),
                      predictions))
        header = 'file_name,actual_class,group_name,score ' + ',score '.join(classes)
        np.savetxt(tmp_dir + 'results/' + split_names[-1] + '.csv', csv, 
                delimiter=',', header=header, comments='', fmt='%s')

        predicted_classes_this_split = np.argmax(predictions, axis=1)
        predicted_classes.extend(predicted_classes_this_split)
        prediction_scores.extend(np.amax(predictions, axis=1))
        accuracy_scores.append(accuracy_score(class_numbers[test], np.argmax(predictions, axis=1)))
        f1_scores = f1_score(
                class_numbers[test], 
                predicted_classes_this_split, 
                average=None, 
                labels=range(len(classes)))
        print('Accuracy:', round(accuracy_scores[-1], 4))
        print('F1 Scores:', f1_scores)
        cm = confusion_matrix(
                class_numbers[test], 
                predicted_classes_this_split,
                labels=np.unique(class_numbers))
        split_metrics[split_name] = [accuracy_scores[-1], f1_scores, cm]
        print_confusion_matrix(cm, classes, normalize=False)
        print()

        # ROC curves, only works for two classes atm
        #fpr, tpr, thresholds = roc_curve(class_numbers[test], predictions[:,1].reshape(-1))
        #tprs.append(interp(mean_fpr, fpr, tpr))
        #tprs[-1][0] = 0.0
        #roc_auc = auc(fpr, tpr)
        #aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3,
        #        #label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))
        #        label='%s (AUC = %0.2f)' % (split_names[-1], roc_auc))
        #if i == 5: break

    if not logo:
        print('--- ', num_folds, '-Fold Cross Validation Results ---', sep='')
    else:
        print('--- Leave One Group Out Cross Validation Results ---', sep='')

    # accuracy
    print("Average Accuracy: %.4f | Standard Deviation: %.4f" % (np.mean(accuracy_scores), np.std(accuracy_scores)))
    if not logo:
        print("Accuracy by fold:", accuracy_scores)
    else:
        print("Accuracy by group:", accuracy_scores)
    #print()

    # f-score
    #cms = [] # can probably remove
    f1s = []
    for key in split_metrics.keys():
        #cms.append(split_metrics[key][2])
        f1s.append(split_metrics[key][1])
    f1s = np.vstack(f1s)
    f1_avgs = np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]), 0, f1s)
    f1_stds = np.apply_along_axis(lambda v: np.std(v[np.nonzero(v)]), 0, f1s)
    f1_avgs[np.isnan(f1_avgs)] = 0.
    f1_stds[np.isnan(f1_stds)] = 0.
    print("Average F1 Score overall: %.4f | Standard Deviation: %.4f" % (np.mean(f1_avgs), np.std(f1_avgs)))
    print("Average F1 Score by class:", f1_avgs)
    print("STD Avg F1 Score by class:", f1_stds)
    print()

    # confusion matrix
    cm = confusion_matrix(actual_classes, predicted_classes)
    print_confusion_matrix(cm, classes)
    print()

    # sanity check, should be the same as above
    #cm = np.zeros(cms[0].shape)
    #for a in cms:
    #    cm = np.add(cm, a)
    #print_confusion_matrix(cm, classes)
    #print()

    #ROC only works for 2 classes atm
    #plt.plot([0,1], [0,1], linestyle='--', lw=2, color='r',
    #        label='Luck', alpha=.8)

    #mean_tpr = np.mean(tprs, axis=0)
    #mean_tpr[-1] = 1.0
    #mean_auc = auc(mean_fpr, mean_tpr)
    #std_auc = np.std(aucs)
    #plt.plot(mean_fpr, mean_tpr, color='b',
    #         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
    #         lw=2, alpha=.8)

    #std_tpr = np.std(tprs, axis=0)
    #tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    #tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    #plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
    #    	     label=r'$\pm$ 1 std. dev.')

    #plt.xlim([-0.05, 1.05])
    #plt.ylim([-0.05, 1.05])
    #plt.xlabel('False Positive Rate')
    #plt.ylabel('True Positive Rate')
    #if logo:
    #    plt.title('ROC Leave One Group Out')
    #else:
    #    plt.title('ROC ' + str(num_folds) + 'Fold Cross Validation')
    ##plt.legend(loc="lower right")
    #plt.legend(loc="best")
    ##plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    #plt.savefig(tmp_dir + '/results/ROC.jpg')

    # data summary by split
    print_class_balance(class_labels, class_numbers,
                        splits, split_names)

    # summarize problem groups 
    summarize_problem_goups = False
    if summarize_problem_goups:
        if logo:
            print("--- Problem Groups (accuracy < 0.7) Summary ---")
            count = 0
            for key in sorted(split_metrics.keys()):
                if split_metrics[key][0] < 0.7:
                    print("Groups Name:", key, "| Accuracy:", round(split_metrics[key][0], 4))
                    print_confusion_matrix(split_metrics[key][2], classes, normalize=False)
                    count += 1
                    print()
            print(count, "problem groups.\n")
        else:
            print("--- Problem Groups (accuracy < 0.7) Summary ---")
            count = 0
            for group in np.unique(group_labels):
                indexes = np.where(group_labels == group)[0]
                score = accuracy_score(np.asarray(actual_classes)[indexes], np.asarray(predicted_classes)[indexes])
                if score < 0.7:
                    print(group, " | count: ", len(indexes), " | accuracy: ", round(score, 4), sep="")
                    count += 1
            print(count, "problem groups.\n")

    # training parameters/config summary
    if summarize_model:
        print_model_info(batch_size, epochs, learning_rate, dropout_rate, model, base_model)
        print()

    # misclassified files
    if summarize_misclassified_images is not None:
        import matplotlib.image as mpimg
        print('--- Misclassified Files ---')

        fig, axes = plt.subplots(25, 1)
        fig.set_size_inches(5, 75)

        #print('file_name predicted_class score')
        misclassified = np.argwhere(np.asarray(actual_classes) != np.asarray(predicted_classes))
        misclassified_indexes = np.random.choice(misclassified.reshape(-1), 25, replace=False)
        for i, axis in enumerate(axes.flat):
            #print(file_names_test[m], classes[predicted_classes[m]], prediction_scores[m]) 
            m = misclassified_indexes[i]
            image_path = data_dir + '/' + file_names_test[m]
            image = mpimg.imread(image_path)
            axis.imshow(image)
            xlabel = "File: %s\nTrue: %s\nPred: %s (%.3f)" % (
                    file_names_test[m], 
                    classes[actual_classes[m]],
                    classes[predicted_classes[m]], 
                    prediction_scores[m])
            axis.set_xlabel(xlabel)

            # Remove ticks from the plot.
            axis.set_xticks([])
            axis.set_yticks([])
            # Remove borders of subplots.
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["bottom"].set_visible(False)
            axis.spines["left"].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(tmp_dir + 'results/misclassified_images.jpg', format='jpg', dpi=300)
        print("random sample of 25 misclassified images saved to", tmp_dir + 'results/misclassified_images.jpg')

        classified = np.argwhere(np.asarray(actual_classes) == np.asarray(predicted_classes))
        classified_indexes = np.random.choice(classified.reshape(-1), 25, replace=False)
        for i, axis in enumerate(axes.flat):
            #print(file_names_test[m], classes[predicted_classes[m]], prediction_scores[m]) 
            m = classified_indexes[i]
            image_path = data_dir + '/' + file_names_test[m]
            image = mpimg.imread(image_path)
            axis.imshow(image)
            xlabel = "File: %s\nTrue: %s\nPred: %s (%.3f)" % (
                    file_names_test[m], 
                    classes[actual_classes[m]],
                    classes[predicted_classes[m]], 
                    prediction_scores[m])
            axis.set_xlabel(xlabel)

            # Remove ticks from the plot.
            axis.set_xticks([])
            axis.set_yticks([])
            # Remove borders of subplots.
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
            axis.spines["bottom"].set_visible(False)
            axis.spines["left"].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(tmp_dir + 'results/classified_images.jpg', format='jpg', dpi=300)
        print("random sample of 25 correctly classified images saved to", tmp_dir + 'results/classified_images.jpg')
        
        csv = tmp_dir + 'results/misclassified_images.csv'
        with open(csv, 'w', newline="") as f:
            f.write('file_name,predicted_class,score\n')
            for m in misclassified.reshape(-1):
                row = file_names_test[m] + ',' + classes[predicted_classes[m]] + ',' + str(prediction_scores[m])
                f.write("%s\n" % row)
        print("csv of all misclassified images saved to", csv)


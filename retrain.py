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

from sklearn.model_selection import train_test_split, StratifiedKFold, GroupShuffleSplit, GroupKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # Suppress TensorFlow message about CPU features


def load_base_model(model_name):
    """Load pre-trained model without final layers."""

    # Need include_top=False and pooling='avg' to generate bottleneck features
    if model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299, 3), pooling='avg')
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    elif model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    else:
        print("Model name not recognized.")
        return
    print('\n' + base_model.name, 'base model with input shape', base_model.input_shape, 'loaded.')
    return base_model


def preprocess_input_wrapper(x):
    """Wrapper around keras.applications.imagenet_utils.preprocess_input()
    to make it compatible for use with keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument.

    Parameters
    ----------
    x : a numpy 3darray (a single image to be preprocessed)

    Note we cannot pass keras.applications.imagenet_utils.preprocess_input()
    directly to to keras.preprocessing.image.ImageDataGenerator's
    `preprocessing_function` argument because the former expects a
    4D tensor whereas the latter expects a 3D tensor. Hence the
    existence of this wrapper.

    Returns a numpy 3darray (the preprocessed image).
    """

    X = np.expand_dims(x, axis=0)
    X = imagenet_utils_preprocess_input(X)
    return X[0]


def create_bottlenecks(bottleneck_file, class_file, data_dir, base_model, groups=None):
    """Returns numpy array of bottlenecks.

    Loads and returns 'bottleneck_file' and 'class_file' as numpy arrays. 
    If 'bottleneck_file' does not exist, features are generated using 'base_model' 
    and a new 'bottleneck_file' and 'class_file' are saved. Note the returned numpy arrays 
    have dtype=string.  When extracting column contents, typecast column contents to 
    the appropriate data type:

        file_names = bottlenecks[:, 0]
        class_numbers = bottlenecks[:, 1].astype(int)
        class_labels = bottlenecks[:, 2]
        group_labels = bottlenecks[:, 3]
        features = bottlenecks[:, 4:].astype(float)

    bottleneck_file column contents:
    file_name(string), class_number(int), class_label(string), group_label(string), features(float)...

    class_file column contents:
    class_label(string), class_number(int)

    Inputs:
        bottleneck_file, class_file: paths to csv files
        data_dir: path to directory of images used to calculate bottlenecks (images are in folders for each class)
        base_model: model used to generate features, it's assumed the model's output is a vector
        img_height, img_width: input dimensions for base_model
        groups (optional): dictionary where key=file_name and value=group_label

    Returns: numpy arrays of 'bottleneck_file' and 'class_file' with dtype=string
    """

    print('Generating bottlenecks...')
    if not os.path.exists(bottleneck_file):
        # Use correct image preprocessing for model
        if base_model.name in ('inception_v3'):
            preprocess_input = inception_v3_preprocess_input
        elif base_model.name in ('vgg16', 'vgg19', 'resnet50'):
            preprocess_input = preprocess_input_wrapper
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
        class_label_names = sorted(images.class_indices.keys()) 

        file_names = images.filenames
        class_numbers = images.classes
        class_labels = []
        for number in class_numbers:
            class_labels.append(class_label_names[number])
        group_labels = []
        if groups is not None:
            for name in file_names:
                group_labels.append(groups[name])
        else:
            for name in file_names:
                group_labels.append('')
        features = base_model.predict_generator(images, images.samples, verbose=1)

        bottlenecks = np.hstack((np.array(file_names).reshape((-1,1)), 
                                 class_numbers.reshape((-1,1)), 
                                 np.array(class_labels).reshape((-1,1)), 
                                 np.array(group_labels).reshape((-1,1)),
                                 features))
        np.savetxt(bottleneck_file, bottlenecks, delimiter=',', fmt='%s')

        class_label_encoding = sorted(images.class_indices.items(), key=lambda x:x[1])
        np.savetxt(class_file, class_label_encoding, delimiter=',', fmt='%s')
        
    else:
        print("Bottlenecks already exist.")
        
    print("Loading bottlenecks...")
    bottlenecks = np.loadtxt(bottleneck_file, delimiter=',', dtype='U')
    classes = np.loadtxt(class_file, delimiter=',', dtype='U')
    print("Bottlenecks loaded.\n")

    return bottlenecks, classes


def create_final_layers(base_model, num_classes, learning_rate=0.001, dropout_rate=0.5):
    """Returns a model that is meant to be trained on features from 'base_model'.
    
    Inputs:
        base_model: model used to generate features, it's assumed the model's output is a vector
        num_classes: int, count of the number of classes in training data
        learning_rate (optional)
        dropout_rate (optional)

    Returns: trainable model 
    """

    # setup final layers using sequential model
    final_layers = Sequential(name='final_layers')
    final_layers.add(Dense(1024, activation='relu', input_shape=base_model.output_shape[1:]))
    final_layers.add(Dropout(dropout_rate))
    final_layers.add(Dense(num_classes, activation='softmax'))

    '''
    # setup final layers using funcional API
    inputs = Input(shape=base_model.output_shape[1:])
    x = Dense(1024, activation='relu')(inputs)
    x = Dropout(0.6)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    final_layers = Model(inputs=inputs, outputs=predictions, name='final_layers')
    '''

    # compile the final layers model
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    final_layers.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return final_layers


def train_and_evaluate_final_layers(
        base_model, bottlenecks, tmp_dir, log_dir, 
        learning_rate=0.001, dropout_rate=0.5, epochs=10, batch_size=16, test_size=0.1,
        save_model=False):
    """Use a train-test split to evaluate final layers in transfer learning. 
    
    Prints training and results summary. If group labels exist, the train-test split will be by group.

    Inputs:
        base_model: model used to generate features, it's assumed the model's output is a vector
        bottlenecks: numpy array of bottlenecks generated by 'create_bottlenecks' function
        tmp_dir: path, trained model is saved here when 'save_model' is True
        log_dir: path, tensorboard logs are saved here
        learning_rate (optional): model hyperparameter
        dropout_rate (optional): model hyperparameter
        epochs (optional): training parameter
        batch_size (optional): training parameter
        test_size (optional): proportion of data to be used for testing
        save_model (optional): if True, the 'base_model' and trained final layers are saved as a complete model in 'tmp_dir'
    """

    #file_names = bottlenecks[:, 0]
    class_numbers = bottlenecks[:, 1].astype(int)
    class_labels = bottlenecks[:, 2]
    group_labels = bottlenecks[:, 3]
    features = bottlenecks[:, 4:].astype(float)

    num_classes = len(set(class_numbers))
    
    # split bottlenecks into train and validation sets
    if group_labels[0] == '':
        train_features, validation_features, train_labels, validation_labels = \
                train_test_split(features, class_numbers, test_size=test_size)
    else:
        train, validate = next(GroupShuffleSplit(n_splits=2, test_size=test_size).split(
                features, class_numbers, group_labels))
        train_features, validation_features, train_labels, validation_labels = \
                features[train], features[validate], \
                class_numbers[train], class_numbers[validate]

    # do one hot encoding for labels
    train_labels_one_hot, validation_labels_one_hot = to_categorical(train_labels), to_categorical(validation_labels)

    print_class_balance(class_labels, class_numbers, 
                        [train_labels, validation_labels], ["Train","Validate"])

    # create final layers model
    final_layers = create_final_layers(
            base_model, num_classes, learning_rate=learning_rate, dropout_rate=dropout_rate)

    # callbacks
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=20, 
          write_graph=False, write_images=False)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                  patience=5, verbose=1)

    #early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    # train final layers
    final_layers.fit(train_features, train_labels_one_hot,
        batch_size=batch_size,
        #callbacks=[reduce_lr, early_stop, tensorboard],
        callbacks=[reduce_lr, tensorboard],
        epochs=epochs,
        shuffle=True,
        validation_data=(validation_features, validation_labels_one_hot),
        verbose=1)

    # training parameters/config summary
    print_model_info(batch_size, epochs, learning_rate, dropout_rate, final_layers)

    # create and save complete retrained model 
    if save_model:
        model = Model(inputs=base_model.input, outputs=final_layers(base_model.output))
        model.save(tmp_dir + base_model.name + '-retrained-model.h5')
        print("\nModel saved to:", tmp_dir + base_model.name + '-retrained-model.h5')
        # capture retrained model architecture with final layers
        save_model_summary(tmp_dir + base_model.name + '-retrained-model-summary.txt', model)


def k_fold_cross_validate(
        base_model, bottlenecks, classes, num_folds=5, 
        learning_rate=0.001, dropout_rate=0.5, epochs=10, batch_size=16,
        summarize_model=True, summarize_misclassified_images=False):
    """Use k-fold cross validation to evaluate final layers in transfer learning. 
    
    Prints validation and results summary. If group labels exist, the folds will split by group.

    Inputs:
        base_model: model used to generate features, it's assumed the model's output is a vector
        bottlenecks: numpy array of bottlenecks generated by 'create_bottlenecks' function
        classes: numpy array of class names/numbers generated by 'create_bottlenecks' function 
        num_folds (optional): number of folds to use
        learning_rate (optional): model hyperparameter
        dropout_rate (optional): model hyperparameter
        epochs (optional): training parameter
        batch_size (optional): training parameter
        summarize_model (optional): prints hyperparamter and model summary
        summarize_misclassified_images (optional): prints list of misclassified images
    """

    file_names = bottlenecks[:, 0]
    class_numbers = bottlenecks[:, 1].astype(int)
    class_labels = bottlenecks[:, 2]
    group_labels = bottlenecks[:, 3]
    features = bottlenecks[:, 4:].astype(float)

    actual_classes = []
    predicted_classes =[]
    prediction_scores = []
    accuracy_scores = []
    file_names_test = []
    folds = []
    fold_names = []
    num_classes = len(set(class_numbers))

    if group_labels[0] == '':
        print('\nPerforming stratified ', num_folds, '-fold cross validation...', sep='')
        kf = StratifiedKFold(n_splits=num_folds, shuffle=True)
    else:
        print('\nPerforming group ', num_folds, '-fold cross validation...', sep='')
        kf = GroupKFold(n_splits=num_folds)
    fold = 1
    # kf.split will ignore group_labels if kf is StratifiedKFold
    for train, test in kf.split(features, class_numbers, group_labels):
        print("Fold ", fold, "/", num_folds, sep='')

        actual_classes.extend(class_numbers[test])
        file_names_test.extend(file_names[test])
        folds.append(class_numbers[test])
        fold_names.append("Fold " + str(fold))
        fold += 1

        model = None # reset the model for each fold
        model = create_final_layers(
                base_model, num_classes, learning_rate=learning_rate, dropout_rate=dropout_rate)
        model.fit(features[train], 
                  to_categorical(class_numbers[train]),
                  batch_size = batch_size,
                  epochs = epochs,
                  shuffle=True,
                  verbose = 2)

        predictions = model.predict(features[test])
        predicted_classes.extend(np.argmax(predictions, axis=1))
        prediction_scores.extend(np.amax(predictions, axis=1))
        accuracy_scores.append(accuracy_score(class_numbers[test], np.argmax(predictions, axis=1)))
        print('Accuracy:', round(accuracy_scores[-1], 4), '\n')

    print_class_balance(class_labels, class_numbers, 
                        folds, fold_names)

    print('--- ', num_folds, '-Fold Cross Validation Results ---', sep='')

    # accuracy
    print("Average Accuracy: %.4f | Standard Deviation: %.4f" % (np.mean(accuracy_scores), np.std(accuracy_scores)))
    print("Accuracy by fold:", accuracy_scores)

    # precision, recall, f-score
    print('\n', classification_report(actual_classes, predicted_classes, target_names=classes[:,0]))

    # confusion matrix
    cm = confusion_matrix(actual_classes, predicted_classes)
    print_confusion_matrix(cm, classes[:,0])

    # training parameters/config summary
    if summarize_model:
        print_model_info(batch_size, epochs, learning_rate, dropout_rate, model)

    # misclassified files
    if summarize_misclassified_images:
        print('\n--- Misclassified Files ---')
        print('file_name predicted_class score')
        misclassified = np.argwhere(np.asarray(actual_classes) != np.asarray(predicted_classes))
        for m in misclassified:
            print(file_names_test[m[0]], classes[:,0][predicted_classes[m[0]]], prediction_scores[m[0]]) 


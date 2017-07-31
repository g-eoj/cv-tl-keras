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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
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
    """Returns dictionary of group membership."""

    print("Loading groups...")
    _ = np.loadtxt(groups_file, delimiter=',', dtype='U')
    groups = {}
    for file_name, group in _:
        groups[file_name] = group
    print(groups_file, "loaded.")

    return groups


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
    print(bottleneck_file, "loaded.")

    return bottlenecks, classes


def create_final_layers(base_model, num_classes, optimizer=None, learning_rate=0.001, dropout_rate=0.5):
    """Returns a model that is meant to be trained on features from 'base_model'.
    
    Inputs:
        base_model: model used to generate features, it's assumed the model's output is a vector
        num_classes: int, count of the number of classes in training data
        optimizer (optional)
        learning_rate (optional)
        dropout_rate (optional)

    Returns: trainable model 
    """

    # setup final layers using sequential model
    final_layers = Sequential(name='final_layers')
    final_layers.add(Dense(
        base_model.output_shape[1] // 2, 
        activation='relu', 
        input_shape=base_model.output_shape[1:]))
    final_layers.add(Dropout(dropout_rate))
    final_layers.add(Dense(num_classes, activation='softmax'))

    '''
    ft = True
    if ft:
        input_shape = None
        model = load_base_model('InceptionV3', input_shape)
        inputs = [model.get_layer('conv2d_73').input, 
                  model.get_layer('conv2d_71').input, 
                  model.get_layer('max_pooling2d_4').input]
        x = model.output
        x = Dense(1024, activation='relu')(inputs)
        x = Dropout(dropout_rate)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        final_layers = Model(inputs=inputs, outputs=predictions, name='final_layers')
    '''


    '''
    # setup final layers using funcional API
    inputs = Input(shape=base_model.output_shape[1:])
    x = Dense(1024, activation='relu')(inputs)
    x = Dropout(dropout_rate)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    final_layers = Model(inputs=inputs, outputs=predictions, name='final_layers')
    '''

    # compile the final layers model
    if optimizer is None:
        optimizer = keras.optimizers.Adam(lr=learning_rate)
    final_layers.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return final_layers


def train_and_evaluate(
        base_model, bottlenecks, tmp_dir, log_dir, test_size=0.1, use_groups=True, use_weights=False,
        optimizer=None, learning_rate=0.001, dropout_rate=0.5, epochs=10, batch_size=16,
        save_model=False):
    """Use a train-test split to evaluate final layers in transfer learning. 
    
    Prints training and results summary. If group labels exist, the train-test split will be by group.

    Inputs:
        base_model: model used to generate features, it's assumed the model's output is a vector
        bottlenecks: numpy array of bottlenecks generated by 'create_bottlenecks' function
        tmp_dir: path, trained model is saved here when 'save_model' is True
        log_dir: path, tensorboard logs are saved here
        test_size (optional): proportion of data to be used for testing
        optimizer (optional): optimizer to use when training final layers
        learning_rate (optional): model hyperparameter
        dropout_rate (optional): model hyperparameter
        epochs (optional): training parameter
        batch_size (optional): training parameter
        save_model (optional): if True, the 'base_model' and trained final layers are saved as a complete model in 'tmp_dir'
    """

    #file_names = bottlenecks[:, 0]
    class_numbers = bottlenecks[:, 1].astype(int)
    class_labels = bottlenecks[:, 2]
    group_labels = bottlenecks[:, 3]
    features = bottlenecks[:, 4:].astype(float)

    num_classes = len(set(class_numbers))
    
    # split bottlenecks into train and validation sets
    if group_labels[0] == '' or not use_groups:
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

    print()
    print_class_balance(class_labels, class_numbers, 
                        [train_labels, validation_labels], ["Train","Validate"])

    # create final layers model
    final_layers = create_final_layers(
            base_model, num_classes, optimizer=optimizer, learning_rate=learning_rate, dropout_rate=dropout_rate)

    # callbacks
    tensorboard = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=0, batch_size=20, 
          write_graph=False, write_images=False)

    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                  patience=5, verbose=1)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

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
    #prediction_scores = np.amax(predictions, axis=1)
    f1_scores = f1_score(
            validation_labels, 
            prediction_labels,
            average=None, 
            labels=range(num_classes))
    print('\nValidation Accuracy:', round(accuracy_score(validation_labels, prediction_labels), 4))
    print('F1 Scores:', f1_scores)
    cm = confusion_matrix(
            validation_labels, 
            prediction_labels,
            labels=range(num_classes))
    print_confusion_matrix(cm, np.unique(class_labels))
    print()

    # training parameters/config summary
    print_model_info(batch_size, epochs, learning_rate, dropout_rate, final_layers)

    # wip - create and save complete retrained model, suitable for fine-tuning 
    if save_model:
        x = base_model.output
        x = Dense(1024, activation='relu')(x)
        x = Dropout(dropout_rate)(x)
        predictions = Dense(num_classes, activation='softmax', name='predictions')(x)
        model = Model(inputs=base_model.input, outputs=predictions)

        for i, layer in enumerate(reversed(final_layers.layers), 1):
            pretrained_weights = layer.get_weights()
            model.layers[-i].set_weights(pretrained_weights)

        #for i, layer in enumerate(base_model.layers):
        #       print(i, layer.name)

        # so last two inception blocks can be fine-tuned
        for layer in model.layers[:249]:
            layer.trainable = False
        for layer in model.layers[249:]:
            layer.trainable = True

        # slow learning rate for fine-tuning
        from keras.optimizers import SGD
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

        model.save(tmp_dir + base_model.name + '-retrained-model.h5')
        print("\nModel saved to:", tmp_dir + base_model.name + '-retrained-model.h5')
        # capture retrained model architecture with final layers
        save_model_summary(tmp_dir + base_model.name + '-retrained-model-summary.txt', model)

    # wip - fine-tuning
    fine_tune = False # temporary
    if fine_tune:

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

        print("Fine-tuning the following layers:")
        model = Model(inputs=base_model.input, outputs=final_layers(base_model.output))
        for i, layer in enumerate(model.layers):
            if i >= 249:
                print(i, layer.name)

        for layer in model.layers[:249]:
            layer.trainable = False
        for layer in model.layers[249:]:
            layer.trainable = True

        from keras.optimizers import SGD
        model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy')

        model.fit(images[train], train_labels_one_hot,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_data=(images[train], validation_labels_one_hot),
            verbose=1)


def group_k_fold(num_folds, features, class_numbers, group_labels):
    """Returns indexes for random folds, has issues if group sizes are imbalanced..."""

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
        base_model, bottlenecks, classes, 
        num_folds=5, logo=False, use_weights=False, resample=None,
        optimizer=None, learning_rate=0.001, dropout_rate=0.5, epochs=10, batch_size=16,
        summarize_model=True, summarize_misclassified_images=False):
    """Use cross validation to evaluate final layers in transfer learning. 
    
    Prints validation and results summary. If group labels exist, the folds will split by group.

    Inputs:
        base_model: model used to generate features, it's assumed the model's output is a vector
        bottlenecks: numpy array of bottlenecks generated by 'create_bottlenecks' function
        classes: numpy array of class names/numbers generated by 'create_bottlenecks' function 
        num_folds (optional): number of folds to use
        logo (optional): do leave one group out cross validation
        use_weights (optional): use class balance to scale the loss function during training
        optimizer (optional): optimizer to use when training final layers
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
            print("Balancing classes in training set with oversampling.")
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
        print("First 10 test files:", sorted(file_names[test])[0:10])
        splits.append(class_numbers[test])
        if not logo:
            split_names.append("Fold " + str(i+1))
        else:
            split_names.append("Group '" + split_name + "'")

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
        predicted_classes_this_split = np.argmax(predictions, axis=1)
        predicted_classes.extend(predicted_classes_this_split)
        prediction_scores.extend(np.amax(predictions, axis=1))
        accuracy_scores.append(accuracy_score(class_numbers[test], np.argmax(predictions, axis=1)))
        f1_scores = f1_score(
                class_numbers[test], 
                predicted_classes_this_split, 
                average=None, 
                labels=classes[:,1])
        print('Accuracy:', round(accuracy_scores[-1], 4))
        print('F1 Scores:', f1_scores)
        cm = confusion_matrix(
                class_numbers[test], 
                predicted_classes_this_split,
                labels=np.unique(class_numbers))
        split_metrics[split_name] = [accuracy_scores[-1], f1_scores, cm]
        print_confusion_matrix(cm, classes[:,0], normalize=False)
        print()
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
    print()

    # f-score
    #cms = [] # can probably remove
    f1s = []
    for key in split_metrics.keys():
        #cms.append(split_metrics[key][2])
        f1s.append(split_metrics[key][1])
    f1s = np.vstack(f1s)
    f1_avgs = np.apply_along_axis(lambda v: np.mean(v[np.nonzero(v)]), 0, f1s)
    f1_avgs[np.isnan(f1_avgs)] = 0.
    print("Average F1 Score overall: %.4f | Standard Deviation: %.4f" % (np.mean(f1_avgs), np.std(f1_avgs)))
    print("Average F1 Score by class:", f1_avgs)
    print()
    #print('\n', classification_report(actual_classes, predicted_classes, target_names=classes[:,0]))

    # confusion matrix
    cm = confusion_matrix(actual_classes, predicted_classes)
    print_confusion_matrix(cm, classes[:,0])
    print()

    # sanity check, should be the same as above
    #cm = np.zeros(cms[0].shape)
    #for a in cms:
    #    cm = np.add(cm, a)
    #print_confusion_matrix(cm, classes[:,0])
    #print()

    # data summary by split
    print_class_balance(class_labels, class_numbers,
                        splits, split_names)

    # summarize problem groups 
    if logo:
        print("--- Problem Groups (accuracy < 0.7) Summary ---")
        count = 0
        for key in sorted(split_metrics.keys()):
            if split_metrics[key][0] < 0.7:
                print("Groups Name:", key, "| Accuracy:", round(split_metrics[key][0], 4))
                print_confusion_matrix(split_metrics[key][2], classes[:,0], normalize=False)
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
    if summarize_misclassified_images:
        print('--- Misclassified Files ---')
        print('file_name predicted_class score')
        misclassified = np.argwhere(np.asarray(actual_classes) != np.asarray(predicted_classes))
        for m in misclassified:
            print(file_names_test[m[0]], classes[:,0][predicted_classes[m[0]]], prediction_scores[m[0]]) 
        print()


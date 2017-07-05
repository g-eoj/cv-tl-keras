import os.path
import sys

from collections import Counter

def print_class_balance(class_labels, class_numbers, 
                        fold_labels, fold_names):
    class_label_names = sorted(set(class_labels))
    columnwidth = max([len(x) for x in class_label_names] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Print header
    print("    " + empty_cell + "--- Class Balance ---".center(
        (columnwidth + 1) * len(class_label_names), ' '))
    print("    " + empty_cell, end=" ")
    for name in class_label_names:
        print("%{0}s".format(columnwidth) % name, end=" ")
    print()

    # Print rows
    for i, fold_name in enumerate(fold_names): 
        print("    %{0}s".format(columnwidth) % fold_name, end=" ")
        counts = Counter(fold_labels[i])
        for j in range(len(class_label_names)):
            if j not in fold_labels[i]:
                proportion = 0
            else:
                proportion = counts[j] #/ float(len(fold_labels[i]))
            cell = "%{0}d".format(columnwidth) % proportion
            print(cell, end=" ")
        print()

    print("    %{0}s".format(columnwidth) % "Total", end=" ")
    counts = Counter(class_numbers)
    for i in range(len(class_label_names)):
        proportion = counts[i] #/ float(len(class_numbers))
        cell = "%{0}d".format(columnwidth) % proportion
        print(cell, end=" ")
    print("\n") 
    

def save_model_summary(filename, model):
    with open(filename, "w") as text_file:
        sys.stdout = text_file
        model.summary()
    sys.stdout = sys.__stdout__


def print_model_info(batch_size, epochs, learning_rate, dropout_rate, model):
    print('\n--- Hyperparameter & Model Summary ---')
    print("Batch Size:", batch_size)
    print("Epochs:", epochs)
    print("Learning Rate:", learning_rate)
    print("Dropout Rate:", dropout_rate)
    print("Optimizer:", model.optimizer)
    print("Optimizer Config:", model.optimizer.get_config())
    #print("Model Config:", model.get_config())
    print("Final Layers:")
    model.summary()


def print_confusion_matrix(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell + "(actual, predicted)".center((columnwidth + 1) * len(labels), ' '))
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

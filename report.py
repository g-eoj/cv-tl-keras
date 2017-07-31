import os.path
from os import listdir
import sys
import numpy as np

from collections import Counter


def group_dict(groups_file):
    """Returns dictionary of group membership."""

    _ = np.loadtxt(groups_file, delimiter=',', dtype='U')
    groups = {}
    for file_name, group in _:
        groups[file_name] = group

    return groups


def data_summary(data_dir, groups=None, verbose=False):
    """
    data_dir: path
    groups: dict
    """
    print("Data Summary:", data_dir)
    class_names = [f for f in listdir(data_dir) if not os.path.isfile(os.path.join(data_dir, f))]
    class_names = sorted(class_names)
    class_width = max([len(x) for x in class_names] + [5])
    grand_total = 0
    if groups is not None:
        group_names = Counter(groups.values()).keys()
        group_width = max([len(x) for x in group_names] + [5])
        group_classes = dict(zip(group_names, [[] for n in group_names]))
        print("%{0}s".format(class_width) % "Class", end="   ")
        print("%{0}s".format(8) % "Group", end="   ")
        print("%{0}s".format(8) % "Count")
        for class_name in class_names:
            print("-"*(2*(8+3)+class_width)) 
            file_names = os.listdir(os.path.join(data_dir, class_name))
            group = []
            for file_name in file_names:
                group.append(groups[class_name + "/" + file_name])
            group_counts = Counter(group)
            for i, key in enumerate(sorted(group_counts.keys())):
                if i == 0:
                    print("%{0}s".format(class_width) % class_name, end="   ")
                else:
                    print("%{0}s".format(class_width) % "", end="   ")
                group_classes[key].append((class_name, group_counts[key]))
                print("%{0}s".format(8) % key, end="   ")
                print("%{0}s".format(8) % group_counts[key])
            print("%{0}s".format(class_width) % "", end="   ")
            print("%{0}s".format(8) % "Total", end="   ")
            print("%{0}s".format(8) % len(file_names))
            grand_total += len(file_names)
        print("-"*(2*(8+3)+class_width)) 
        print("Grand total:", grand_total)

        if verbose:
            group_counts = Counter(groups.values())
            counts = list(group_counts.values())
            print("\nGroup Summary:", len(group_names), "groups found | median size:", 
                    round(np.median(counts), 2), "| mad:", round(np.median(np.abs(np.array(counts) - np.median(counts)))), 
                    "| mean size:", round(np.mean(counts), 2), "| sd:", round(np.std(counts), 2))
            print("%{0}s".format(group_width) % "Group", end="   ")
            print("Class Membership\\Counts")
            print("-"*(1*(23+3)+group_width)) 
            for group_name in sorted(group_names):
                print("%{0}s".format(group_width) % group_name, end="   ")
                print(group_classes[group_name], "Total:", group_counts[group_name])
    else:
        print("%{0}s".format(class_width) % "Class", end="   ")
        print("%{0}s".format(8) % "Count")
        for class_name in class_names:
            print("-"*(1*(8+3)+class_width)) 
            file_names = os.listdir(os.path.join(data_dir, class_name))
            print("%{0}s".format(class_width) % class_name, end="   ")
            print("%{0}s".format(8) % len(file_names))
            grand_total += len(file_names)
            print("-"*(1*(8+3)+class_width)) 
        print("Grand total:", grand_total)
    print()


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


def print_model_info(batch_size, epochs, learning_rate, dropout_rate, model, base_model=None):
    print('--- Hyperparameter & Model Summary ---')
    if base_model is not None:
        print("Base Model:", base_model.name)
        print("Feature Layer:", base_model.layers[-2].name)
    print("Batch Size:", batch_size)
    print("Epochs:", epochs)
    #print("Learning Rate:", learning_rate)
    print("Dropout Rate:", dropout_rate)
    print("Optimizer:", model.optimizer)
    print("Optimizer Config:", model.optimizer.get_config())
    #print("Model Config:", model.get_config())
    print("Final Layers:")
    model.summary()


def print_confusion_matrix(cm, labels,
                           hide_zeroes=False, hide_diagonal=False, hide_threshold=None,
                           normalize=True):
    """pretty print for confusion matrixes"""
    if normalize:
        np.seterr(divide='ignore', invalid='ignore')
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell + "--- Confusion Matrix (actual, predicted) ---".center((columnwidth + 1) * len(labels), ' '))
    #print("    " + empty_cell + "(actual, predicted)".center((columnwidth + 1) * len(labels), ' '))
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            if normalize:
                cell = "%{0}.2f".format(columnwidth) % cm[i, j]
            else:
                cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def getROC(ground_truth, scores, class_list):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    ground_truth = np.array(ground_truth);
    scores = np.array(scores);
    scores = np.squeeze(scores)
    for i in range(len(class_list)):
        (fpr[i], tpr[i], _) = roc_curve(ground_truth[:, i], scores[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    return (fpr, tpr, roc_auc)


def multiClassROC(fpr, tpr, roc_auc, class_list,):
    n_classes = len(class_list)
    lw = 2  # line width

    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    # Plot all ROC curves
    # fig = plt.figure()

    plt.plot(
        fpr['macro'],
        tpr['macro'],
        label='macro (a = {0:0.2f})'.format(roc_auc['macro']),
        color='navy',
        linestyle=':',
        linewidth=4,
        )

    colors = cycle([
        'aqua',
        'darkorange',
        'cornflowerblue',
        'green',
        'red',
        'blue',
        'black',
        ])
    for (i, color) in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='{0} (a={1:0.2f})'.format(class_list[i],
                 roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC plot')
    plt.legend(loc='lower right')
    plt.savefig('test_ROC.png')


# plt.show()

'''
def save_roc(actual_classes, 
    class_list = sorted(set(class_labels))
    (fpr, tpr, roc_auc) = getROC(truth, prediction_scores_list, class_list)
    multiClassROC(fpr, tpr, roc_auc, class_list)
'''

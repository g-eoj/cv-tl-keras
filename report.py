import os.path
from os import listdir
import sys
import numpy as np

from collections import Counter


class Logger(object):
    """Saves terminal output to log file."""

    def __init__(self, path):
        self.terminal = sys.stdout
        self.log = open(path + 'log.txt', 'w')

    def isatty(self):
        return self.terminal.isatty()

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message.replace('\b', '').replace('\r', '\n'))

    def flush(self):
        pass


def group_dict(groups_file):
    """Returns dictionary of group membership, where the keys are file names.

    input: file path to csv which have rows with the format: file_name,group
    """

    _ = np.loadtxt(groups_file, delimiter=',', dtype='U')
    groups = {}
    for file_name, group in _:
        groups[file_name] = group

    return groups


def data_summary(data_dir, groups_file=None, csv=None):
    """Summarizes counts for images in data_dir.

    Inputs:
        data_dir: path to directory of images (where images are in a 
            subdirectory for each class)
        groups_file (optional): file path to csv which have rows with the 
            format: file_name,group
        csv (optional): file path to save csv of counts, if groups_file exists
    """

    print("Data Summary:", data_dir, "\n")
    class_names = [f for f in listdir(data_dir) if not os.path.isfile(os.path.join(data_dir, f))]
    class_names = sorted(class_names)
    class_width = max([len(x) for x in class_names] + [8])
    grand_total = 0
    if groups_file is not None:
        groups = group_dict(groups_file)
        group_names = sorted(Counter(groups.values()).keys())
        group_width = max([len(x) for x in group_names] + [5])
        groups_discovered = set() # needed since it's possible to use a groups_file 
                                  # that lists groups not used in the data_dir
        print('{0:>{1}}'.format("Class", class_width), end="   ")
        print('{0:>{1}}'.format("Group", group_width), end="")
        print('{0:>8}'.format("Count"))
        for class_name in class_names:
            print("-"*(class_width+group_width+3+8)) 
            file_names = os.listdir(os.path.join(data_dir, class_name))
            group = []
            for file_name in file_names:
                group.append(groups[class_name + "/" + file_name])
            group_counts = Counter(group)
            groups_discovered |= set(group)
            for i, group_name in enumerate(sorted(group_counts.keys())):
                if i == 0:
                    print('{0:>{1}}'.format(class_name, class_width), end="   ")
                else:
                    print('{0:>{1}}'.format("", class_width), end="   ")
                print('{0:>{1}}'.format(group_name, group_width), end="")
                print('{0:>8}'.format(group_counts[group_name]))
            print('{0:>{1}}'.format("", class_width), end="   ")
            print('{0:>{1}}'.format("Total", group_width), end="")
            print('{0:>8}'.format(len(file_names)))
            grand_total += len(file_names)
        print("-"*(class_width+group_width+3+8)) 

        rows = []
        class_totals = []
        group_totals = {}
        row = "class,"
        for group_name in group_names:
            row += group_name + ","
            group_totals[group_name] = 0
        row += "total"
        rows.append(row)
        for class_name in class_names:
            row = class_name + "," 
            total = 0
            group = []
            file_names = os.listdir(os.path.join(data_dir, class_name))
            for file_name in file_names:
                group.append(groups[class_name + "/" + file_name])
            group_counts = Counter(group)
            for group_name in group_names:
                if group_name in group_counts.keys(): 
                    row += str(group_counts[group_name]) + ","
                    total += group_counts[group_name]
                    group_totals[group_name] += group_counts[group_name]
                else:
                    row += str(0) + ","
            row += str(total)
            class_totals.append(total)
            rows.append(row)
        row = "total,"
        for group_name in group_names:
            row += str(group_totals[group_name]) + ","
        row += str(grand_total)
        rows.append(row)

        print(len(class_names), "classes |", 
              len(groups_discovered), "groups |",
              grand_total, "images") 
        counts = list(group_totals.values())
        print("median class size:", round(np.median(class_totals), 2), 
              "| mad:", round(np.median(
                  np.abs(np.array(class_totals) - np.median(class_totals))))) 
        print("mean class size:", round(np.mean(class_totals), 2), 
              "| sd:", round(np.std(class_totals), 2))
        print("median group size:", round(np.median(counts), 2), 
              "| mad:", round(np.median(
                  np.abs(np.array(counts) - np.median(counts))))) 
        print("mean group size:", round(np.mean(counts), 2), 
              "| sd:", round(np.std(counts), 2))
        print()

        # csv
        for row in rows:
            print(row)
        if csv is not None:
            with open(csv, 'w', newline="") as f:
                for row in rows:
                    f.write("%s\n" % row)
            print("csv saved to", csv)
    else:
        print('{0:>{1}}'.format("Class", class_width), end="   ")
        print('{0:>8}'.format("Count"))
        print("-"*(class_width+3+8)) 
        class_totals =[]
        for class_name in class_names:
            file_names = os.listdir(os.path.join(data_dir, class_name))
            print('{0:>{1}}'.format(class_name, class_width), end="   ")
            print('{0:>8}'.format(len(file_names)))
            class_totals.append(len(file_names))
            grand_total += len(file_names)
        print("-"*(class_width+3+8)) 
        print(len(class_names), "classes |", 
              grand_total, "images") 
        print("median class size:", round(np.median(class_totals), 2), 
              "| mad:", round(np.median(
                  np.abs(np.array(class_totals) - np.median(class_totals))))) 
        print("mean class size:", round(np.mean(class_totals), 2), 
              "| sd:", round(np.std(class_totals), 2))
    print()


def print_class_balance(class_labels, class_numbers, 
                        fold_labels, fold_names):
    class_label_names = sorted(set(class_labels))
    first_column_width = max([len(x) for x in fold_names] + [5])  # 5 is value length
    columnwidth = max([len(x) for x in class_label_names] + [5])  # 5 is value length
    empty_cell = " " * first_column_width

    # Print header
    print("    " + empty_cell + "--- Class Balance ---".center(
        (columnwidth + 1) * len(class_label_names), ' '))
    print("    " + empty_cell, end=" ")
    for name in class_label_names:
        print("%{0}s".format(columnwidth) % name, end=" ")
    print()

    # Print rows
    for i, fold_name in enumerate(fold_names): 
        print("    %{0}s".format(first_column_width) % fold_name, end=" ")
        counts = Counter(fold_labels[i])
        for j in range(len(class_label_names)):
            if j not in fold_labels[i]:
                proportion = 0
            else:
                proportion = counts[j] #/ float(len(fold_labels[i]))
            cell = "%{0}d".format(columnwidth) % proportion
            print(cell, end=" ")
        print()

    print("    %{0}s".format(first_column_width) % "Total", end=" ")
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


def print_model_info(batch_size, epochs, model, optimizer, base_model=None):
    print('--- Hyperparameter & Model Summary ---')
    if base_model is not None:
        print("Base Model:", base_model.name)
        print("Feature Extraction Layer:", base_model.layers[-1].name)
    print("Batch Size:", batch_size)
    print("Epochs:", epochs)
    print("Optimizer:", optimizer)
    print("Optimizer Config:", optimizer.get_config())
    print("Model Config:", model.get_config())
    print("Model Layers:")
    model.summary()
    print()

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

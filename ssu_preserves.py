from keras import backend as K
from retrain import create_bottlenecks, k_fold_cross_validate, \
        load_base_model, train_and_evaluate_final_layers

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
data_dir = './research/ssu_preserves/training_images'
tmp_dir = './research/ssu_preserves/tmp/'
log_dir = tmp_dir + 'logs/'
bottleneck_file = tmp_dir + base_model.name + '-bottlenecks.txt'
class_file = tmp_dir + base_model.name + '-retrained-classes.txt'
groups_file = None # csv -> file_name,group

# get bottlenecks and classes
#groups = group_dict(groups_file)
groups = None
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
            learning_rate=0.001, dropout_rate=0.6, epochs=20, batch_size=20,
            summarize_model=True, summarize_misclassified_images=True)

K.clear_session() # prevent TensorFlow error message


from operator import sub
import os
from random import sample
import shutil
from datetime import datetime
import subprocess

class models_genesis_config:
    model = 'Vnet'
    suffix = 'genesis_stroke_cta'
    exp_name = model + '-' + suffix

    input_channels = 1
    input_rows = 128
    input_cols = 128
    input_deps = 128
    nb_class = 1

    resolution_string = (
        str(input_rows) + 'x'
        + str(input_cols) + 'x'
        + str(input_deps)
    )

    vol_path = (
        '/mnt/'
        + 'vol_'
        + resolution_string
    )

    # data
    data = os.path.join(vol_path, 'data')
    data_train = os.path.join(data, 'train')
    data_val = os.path.join(data, 'val')
    originals = os.path.join(data, 'originals')
    segmentations = os.path.join(data, 'segmentations')

    test_segmentation_data = os.path.join(data, 'test_segmentations')
    originals_train = os.path.join(test_segmentation_data, 'train')
    originals_val = os.path.join(test_segmentation_data, 'val')
    segmentations_train = os.path.join(test_segmentation_data, 'train')
    segmentations_val = os.path.join(test_segmentation_data, 'val')
    
    # hyperparameter
    verbose = 1
    weights = None
    batch_size = 1
    activate = 'softmax'
    optimizer = 'adam'
    loss = 'binary_crossentropy'
    metrics = ['acc']
    nb_epoch = 1000
    patience = 50
    lr = 1e-2
    mixed_precision = False

    # logs
    timestamp = datetime.now()
    model_path = (
        vol_path
        + '/seg_results'
        + '/training_'
        + str(timestamp)
    )
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, 'logs')
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    
    # capture git commit hash for replicability
    path_repo = os.path.join(vol_path, 'nageler_doctorate')
    cmd_repo = 'git -C ' + path_repo + ' rev-parse HEAD'
    commit_repo_object = subprocess.run(cmd_repo, capture_output=True, text=True, shell=True)
    commit_repo = commit_repo_object.stdout
    cmd_submodule = 'git rev-parse HEAD'
    commit_submodule_object = subprocess.run(cmd_submodule, capture_output=True, text=True, shell=True)
    commit_submodule = commit_submodule_object.stdout

    def display(self):
        print('\nConfigurations:')
        for a in dir(self):
            if not a .startswith('__') and not callable(getattr(self, a)):
                print('{:30} {}'.format(a, getattr(self, a)))
        print('\n')
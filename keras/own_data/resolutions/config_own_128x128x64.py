import os
import shutil
from datetime import datetime
import subprocess

class models_genesis_config:
    model = "Vnet"
    suffix = "genesis_stroke_cta"
    exp_name = model + "-" + suffix
    
    # train_fold=[0,1,2,3,4]
    # valid_fold=[5,6]
    # test_fold=[7,8,9]
    # hu_min = -1000.0
    # hu_max = 1000.0
    # scale = 32

    input_rows = 128
    input_cols = 128
    input_deps = 64
    nb_class = 1

    resoltuion_string = (
        str(input_rows) + 'x'
        + str(input_cols) + 'x'
        + str(input_deps)
    )
    
    vol_path = (
        '/mnt/'
        + 'vol_'
        + resoltuion_string
    )
    # data
    data = os.path.join(vol_path, 'data')
    data_train = os.path.join(data, 'train')
    data_val = os.path.join(data, 'val')

    # model pre-training
    verbose = 1
    weights = None
    batch_size = 3
    optimizer = "sgd"
    workers = 8
    max_queue_size = workers * 4
    save_samples = "png"
    nb_epoch = 10000
    patience = 50
    lr = 1e0

    # image deformation
    nonlinear_rate = 0.9
    paint_rate = 0.9
    outpaint_rate = 0.8
    inpaint_rate = 1.0 - outpaint_rate
    local_rate = 0.5
    flip_rate = 0.4
    
    # logs
    timestamp = datetime.now()
    model_path = (
        vol_path
        + '/pretrain_results'
        + '/pretraining_'
        + str(timestamp))

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    logs_path = os.path.join(model_path, "Logs")
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)
    sample_path = "pair_samples"
    if not os.path.exists(sample_path):
        os.makedirs(sample_path)
    shutil.rmtree(os.path.join(sample_path, exp_name), ignore_errors=True)
    if not os.path.exists(os.path.join(sample_path, exp_name)):
        os.makedirs(os.path.join(sample_path, exp_name))

    # capture git commit hash for replicability
    path_repo = os.path.join(vol_path, 'nageler_doctorate')
    cmd_repo = 'git -C ' + path_repo + ' rev-parse HEAD'
    commit_repo_object = subprocess.run(cmd_repo, capture_output=True, text=True, shell=True)
    commit_repo = commit_repo_object.stdout
    cmd_submodule = 'git rev-parse HEAD'
    commit_submodule_object = subprocess.run(cmd_submodule, capture_output=True, text=True, shell=True)
    commit_submodule = commit_submodule_object.stdout
    
    def display(self):
        """Display Configuration values."""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")

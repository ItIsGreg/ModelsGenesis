from __future__ import print_function
from lib2to3.pgen2.token import OP

from optparse import OptionParser
import importlib

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import keras
print("keras = {}".format(keras.__version__))
import tensorflow as tf
print("tensorflow-gpu = {}".format(tf.__version__))
try:
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
except:
    pass

import shutil
import numpy as np
from tqdm import tqdm
import pickle

from ModelsGenesis.keras.utils import *
from ModelsGenesis.keras.unet3d import *
from keras.callbacks import LambdaCallback,TensorBoard,ReduceLROnPlateau

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Specify config from CLI
# Leave this script as is
parser = OptionParser()
parser.add_option('--config', dest='cli_config', default=None, type='str')
(options,args) = parser.parse_args()
cli_config = options.cli_config
cli_config = importlib.import_module(cli_config)
conf = cli_config.models_genesis_config()
conf.display()

# Save config for replicability
outfile = open(os.path.join(conf.model_path, 'config'), 'wb')
pickle.dump(conf, outfile)
outfile.close()

# Initialize GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

# Load data
train_files = os.listdir(conf.data_train)
val_files = os.listdir(conf.data_val)
train_paths = [os.path.join(conf.data, file) for file in train_files]
val_paths = [os.path.join(conf.data, file) for file in val_files]

x_train = []
for path in train_paths:
    s = np.load(path)
    x_train.append(s)
x_train = np.expand_dims(np.array(x_train), axis=1)

x_valid = []
for path in val_paths:
    s = np.load(path)
    x_valid.append(s)
x_valid = np.expand_dims(np.array(x_valid), axis=1)

print("x_train: {} | {:.2f} ~ {:.2f}".format(x_train.shape, np.min(x_train), np.max(x_train)))
print("x_valid: {} | {:.2f} ~ {:.2f}".format(x_valid.shape, np.min(x_valid), np.max(x_valid)))

# Setup Model
if conf.model == "Vnet":
    model = unet_model_3d((1, conf.input_rows, conf.input_cols, conf.input_deps), batch_normalization=True)
if conf.weights is not None:
    print("Load the pre-trained weights from {}".format(conf.weights))
    model.load_weights(conf.weights)
model.compile(optimizer=keras.optimizers.SGD(lr=conf.lr, momentum=0.9, decay=0.0, nesterov=False), 
              loss="MSE", 
              metrics=["MAE", "MSE"])

if os.path.exists(os.path.join(conf.model_path, conf.exp_name+".txt")):
    os.remove(os.path.join(conf.model_path, conf.exp_name+".txt"))
with open(os.path.join(conf.model_path, conf.exp_name+".txt"),'w') as fh:
    model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))

shutil.rmtree(os.path.join(conf.logs_path, conf.exp_name), ignore_errors=True)
if not os.path.exists(os.path.join(conf.logs_path, conf.exp_name)):
    os.makedirs(os.path.join(conf.logs_path, conf.exp_name))
tbCallBack = TensorBoard(log_dir=os.path.join(conf.logs_path, conf.exp_name),
                         histogram_freq=0,
                         write_graph=True, 
                         write_images=True,
                        )
tbCallBack.set_model(model)

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                               patience=conf.patience, 
                                               verbose=0,
                                               mode='min',
                                              )
check_point = keras.callbacks.ModelCheckpoint(os.path.join(conf.model_path, conf.exp_name+".h5"),
                                              monitor='val_loss', 
                                              verbose=1, 
                                              save_best_only=True, 
                                              mode='min',
                                             )
lrate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10,
                                    min_delta=0.0001, min_lr=1e-6, verbose=1)

callbacks = [check_point, early_stopping, tbCallBack, lrate_scheduler]

while conf.batch_size > 1:
    # To find a largest batch size that can be fit into GPU
    try:
        model.fit_generator(generate_pair(x_train, conf.batch_size, config=conf, status="train"),
                            validation_data=generate_pair(x_valid, conf.batch_size, config=conf, status="test"), 
                            validation_steps=x_valid.shape[0]//conf.batch_size,
                            steps_per_epoch=x_train.shape[0]//conf.batch_size, 
                            epochs=conf.nb_epoch,
                            max_queue_size=conf.max_queue_size, 
                            workers=conf.workers, 
                            use_multiprocessing=True, 
                            shuffle=True,
                            verbose=conf.verbose, 
                            callbacks=callbacks,
                            )
        break
    except tf.errors.ResourceExhaustedError as e:
        conf.batch_size = int(conf.batch_size - 2)
        print("\n> Batch size = {}".format(conf.batch_size))
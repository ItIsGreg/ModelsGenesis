from calendar import c
from email.policy import default
import imp
from pydoc import cli
import keras
from matplotlib.pyplot import axis
from sklearn.utils import shuffle
from ModelsGenesis.keras.unet3d import *
from keras.layers import GlobalAveragePooling3D, Dense
from keras.callbacks import LambdaCallback, TensorBoard, ReduceLROnPlateau
from keras.utils import to_categorical
from preprocessing.utils import split_by_cutoff
import os
import shutil
import tensorflow as tf
from optparse import OptionParser
import importlib
import pickle
import pandas as pd

##################
# OPTION PARSING v
##################
# specify config from CLI
# leave this script as is
parser = OptionParser()
parser.add_option('--config', dest='cli_config', default=None, type='str')
(options, args) = parser.parse_args()
cli_config = options.cli_config
cli_config = importlib.import_module(cli_config)
conf = cli_config.models_genesis_config()
conf.display()

# save config for replicability
outfile = open(os.path.join(conf.model_path, 'config'), 'wb')
pickle.dump(conf, outfile)
outfile.close()
##################
# OPTION PARSING ^
##################

##################
# GPU SETUP v
##################
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUS,', len(logical_gpus), 'Locial GPUs')
    except RuntimeError as e:
        print(e)
##################
# GPU SETUP ^
##################

##################
# DATA LOADING v
##################
print('Start loading data...')

db_extract = pd.read_excel(conf.db_extract)

train_0_paths, train_1_paths = split_by_cutoff(cutoff=conf.cutoff, src_dir=conf.data_train, db_extract=db_extract)
val_0_paths, val_1_paths = split_by_cutoff(cutoff=conf.cutoff, src_dir=conf.data_val, db_extract=db_extract)

label_train_0 = np.array([0 for _ in range(len(train_0_paths))])
label_train_1 = np.array([1 for _ in range(len(train_1_paths))])
label_val_0 = np.array([0 for _ in range(len(val_0_paths))])
label_val_1 = np.array([1 for _ in range(len(val_1_paths))])
label_train = np.concatenate((label_train_0, label_train_1), axis=0)
label_val = np.concatenate((label_val_0, label_val_1), axis=0)

train_paths = train_0_paths + train_1_paths
val_paths = val_0_paths + val_1_paths

X = np.array([np.load(scan) for scan in train_paths])
x_val = np.array([np.load(scan) for scan in val_paths])
X = np.expand_dims(X, axis=1)
x_val = np.expand_dims(x_val, axis=1)

y = to_categorical(label_train)
y_val = to_categorical(label_val)

print('Shape X: ' + str(X.shape))
print('Shape y: ' + str(y.shape))
print('Shape x_val: ' + str(x_val.shape))
print('Shape y_val: ' + str(y_val.shape))

validation_data = (x_val, y_val)
##################
# DATA LOADING ^
##################

##################
# MODEL SETUP v
##################
models_genesis = unet_model_3d(input_shape=(conf.input_channels ,conf.input_rows, conf.input_cols, conf.input_deps), batch_normalization=True)
if conf.weights is not None:
    print('Load pre-trained Models Genesis weights from {}'.format(conf.weights))
    models_genesis.load_weights(conf.weights)

x = models_genesis.get_layer('depth_7_relu').output
x = GlobalAveragePooling3D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(conf.nb_class, activation=conf.activate)(x)
model = keras.models.Model(inputs=models_genesis.input, outputs=output)

model.compile(optimizer=conf.optimizer, loss=conf.loss, metrics=conf.metrics)
##################
# MODEL SETUP ^
##################

########################
# LOGGING & CALLBACKS v
########################
if os.path.exists(os.path.join(conf.model_path, conf.exp_name+'.txt')):
    os.remove(os.path.join(conf.model_path, conf.exp_name+'.txt'))
with open(os.path.join(conf.model_path, conf.exp_name+'.txt'), 'w') as fh:
    model.summary(positions=[.3, .55, .67, 1.], print_fn=lambda x: fh.write(x + '\n'))

shutil.rmtree(os.path.join(conf.logs_path, conf.exp_name), ignore_errors=True)
if not os.path.exists(os.path.join(conf.logs_path, conf.exp_name)):
    os.makedirs(os.path.join(conf.logs_path, conf.exp_name))

tbCallBack = TensorBoard(
    log_dir=os.path.join(conf.logs_path, conf.exp_name),
    histogram_freq=0,
    write_graph=True,
    write_images=True,
)
tbCallBack.set_model(model)

early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=conf.patience,
    verbose=0,
    mode='min',
    )

check_point = keras.callbacks.ModelCheckpoint(
    os.path.join(conf.model_path, conf.exp_name+'.h5'),
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

lrate_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,
    min_delta=0.0001,
    min_lr=1e-6,
    verbose=1
)
callbacks = [check_point, early_stopping, tbCallBack, lrate_scheduler]
########################
# LOGGING & CALLBACKS ^
########################

while conf.batch_size > 1:
    try:
        model.fit(
            X,y,
            batch_size=conf.batch_size,
            epochs=conf.nb_epoch,
            shuffle=True,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=conf.verbose
            )
        break
    except tf.errors.ResourceExhaustedError as e:
        conf.batch_size = int(conf.batch_size - 2)
        print('\n> Batch size = {}'.format(conf.batch_size))
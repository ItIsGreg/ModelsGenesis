from email.policy import default
import imp
from pydoc import cli
import keras
from ModelsGenesis.keras.unet3d import *
from keras.layers import GlobalAveragePooling3D, Dense
from keras.callbacks import LambdaCallback, TensorBoard, ReduceLROnPlateau
from keras.utils import to_categorical
from preprocessing import data_container as dc
import os
import tensorflow as tf
from optparse import OptionParser
import importlib
import pickle


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

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), 'Physical GPUS,', len(logical_gpus), 'Locial GPUs')
    except RuntimeError as e:
        print(e)

data = dc.data_container()

########################
# MOVE TO CONFIG
########################
base_path = ''
filename_x_train = ''
filename_y_train = ''
filename_x_val = ''
filename_y_val = ''

path_x_train = os.path.join()
path_y_train = os.path.join()
path_x_val = os.path.join()
path_y_val = os.path.join()
########################

# load in preprocessed files
data.load_preprocessed(
    path_x_train,
    path_y_train,
    path_x_val,
    path_y_val
)

# make labels one hot encoded
data.y_train = to_categorical(data.y_train)
data.y_val = to_categorical(data.y_val)

# add a dimension 
data.x_train = np.expand_dims(data.x_train, axis=1)
data.y_train = np.expand_dims(data.x_val, axis=1)

X = data.x_train
y = data.y_train
validation_data = (data.x_val, data.y_val)


########################
# LOGGING
########################

########################
# CALLBACKS
########################


models_genesis = unet_model_3d(input_shape=(conf.input_rows, conf.input_cols, conf.input_deps), batch_normalization=True)
if conf.weights is not None:
    print('Load pre-trained Models Genesis weights from {}'.format(conf.weights))
    models_genesis.load_weights(conf.weights)

x = models_genesis.get_layer('depth_7_relu').output
x = GlobalAveragePooling3D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(conf.nb_class, activation=conf.activate)(x)
model = keras.models.Model(inputs=models_genesis.input, outputs=output)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
model.fit(X,y, batch_size='', epochs='', shuffle=True, validation_data=validation_data)
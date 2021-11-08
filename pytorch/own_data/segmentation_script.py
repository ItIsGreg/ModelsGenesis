from sched import scheduler
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import ModelsGenesis.pytorch.unet3d as unet3d
from ModelsGenesis.pytorch.own_data.SegDataset import SegDataset
import numpy as np
from optparse import OptionParser
import importlib
import pickle
import os
from torchsummary import summary
import sys

##################
# OPTION PARSING v
##################
# specify config from cli
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

print('torch = {}'.format(torch.__version__))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

##################
# DICE LOSS v
##################
def torch_dice_coef_loss(y_true, y_pred, smooth=1.):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    intersection = torch.sum(y_true_f * y_pred_f)
    return 1. - ((2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth))

##################
# DICE LOSS ^
##################

##################
# DATA LOADING v
##################
training_dataset = SegDataset(conf.originals_train, conf.segmentations_train)
validation_dataset = SegDataset(conf.originals_val, conf.segmentations_val)

train_loader = DataLoader(training_dataset, batch_size=conf.batch_size, shuffle=True)
validation_loader = DataLoader(validation_dataset, batch_size=conf.batch_size, shuffle=True)

##################
# DATA LOADING ^
##################


##################
# MODEL SETUP v
##################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = unet3d.UNet3D()

if conf.weights != None:
    weight_dir = conf.weights
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['state_dict']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace('module.', '')] = state_dict[key]
    model.load_state_dict(unParalled_state_dict)
    print('Loading weights from ', conf.weights)

model = nn.DataParallel(model, device_ids=[i for i in range(torch.cuda.device_count())])
model.to(device)

print('Total CUDA devices: ', torch.cuda.device_count())

summary(model, (1, conf.input_rows,conf.input_cols, conf.input_deps), batch_size=conf.batch_size)

criterion = torch_dice_coef_loss
optimizer = torch.optim.SGD(model.parameters(), conf.lr, momentum=0.9, weight_decay=0.0, nesterov=False)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# to track the training loss as the model trains
train_losses = []
# to track the validation loss as the model trains
valid_losses = []
# to track the average training loss per epoch as the model trains
avg_train_losses = []
# to track the average validation loss per epoch as the model trains
avg_valid_losses = []
best_loss = 100000
initial_epoch = 0
num_epoch_no_improvement = 0
sys.stdout.flush()
##################
# MODEL SETUP ^
##################


##################
# TRAIN LOOP v
##################

for epoch in range(initial_epoch, conf.nb_epoch):
    scheduler.step(epoch)
    model.train()
    for batch_ndx, (x,y) in enumerate(train_loader):
        x, y = x.float().to(device), y.float().to(device)
        
        if conf.mixed_precision:
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = criterion(pred, y)
        else:
            pred = model(x)
            loss = criterion(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(round(loss.item(), 2))
        if(batch_ndx + 1) % 5 ==0:
            print('Epoch [{}/{}], iteration {}, Loss{:.06f}'
                .format(epoch + 1, conf.nb_epoch, batch_ndx +1, np.average(train_losses)))
            sys.stdout.flush()
    
    with torch.no_grad():
        model.eval()
        print('validatig...')
        for batch_ndx, (x,y) in enumerate(validation_loader):
            x,y = x.float().to(device), y.float().to(device)
            pred = model(x)
            loss = criterion(pred, y)
            valid_losses.append(round(loss.item(),2))

    ##################
    # LOGGING v
    ##################
    train_loss = np.average(train_losses)
    valid_loss = np.average(valid_losses)
    avg_train_losses.append(train_loss)
    avg_valid_losses.append(valid_loss)
    print('Epoch {}, validation loss is {:.4f}, training loss is {:.4f}.'.format(epoch+1, valid_loss, train_loss))
    train_losses = []
    valid_losses = []
    if valid_loss < best_loss:
        print('Validation loss decreases from {:.4f} to {:.4f}'.format(best_loss, valid_loss))
        best_loss = valid_loss
        num_epoch_no_improvement = 0
        # save model
        torch.save({
            'epoch': epoch+1,
            'state_dict' : model.state_dict(),
            'optimizer_state_dict' : optimizer.state_dict()
        }, os.path.join(conf.model_path, 'Genesis_Chest_CT.pt'))
        print('Saving model ', os.path.join(conf.model_path, 'Genesis_Chest_CT.pt'))
    else:
        print('Validation loss does not decrease from {:.4f}, num_epoch_no_improvement {}'.format(best_loss, num_epoch_no_improvement))
        num_epoch_no_improvement += 1
    if num_epoch_no_improvement == conf.patience:
        print('Early Stopping')
        break
    sys.stdout.flush()
    ##################
    # LOGGING ^
    ##################

##################
# TRAIN LOOP ^
##################
run_flag = 1

import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import sklearn.ensemble
import scipy.stats
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import xarray as xr
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import sklearn.ensemble
import scipy.stats
from sklearn.model_selection import train_test_split 
import torch
#import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import sys
import os

def plot_nino_time_series(y, predictions, title):
  """
  inputs
  ------
    y           pd.Series : time series of the true Nino index
    predictions np.array  : time series of the predicted Nino index (same
                            length and time as y)
    titile                : the title of the plot

  outputs
  -------
    None.  Displays the plot
  """
  predictions = pd.Series(predictions, index=y.index)
  predictions = predictions.sort_index()
  y = y.sort_index()

  plt.plot(y, label='Ground Truth')
  plt.plot(predictions, '--', label='ML Predictions')
  plt.legend(loc='best')
  plt.title(title)
  plt.ylabel('Nino3.4 Index')
  plt.xlabel('Date')
  plt.show()
  plt.close()

class AMV_dataset(Dataset):
    def __init__(self, predictors, predictands):
        self.predictors = predictors
        self.predictands = predictands
        assert self.predictors.shape[0] == self.predictands.shape[0], \
               "The number of predictors must equal the number of predictands!"

    def __len__(self):
        return self.predictors.shape[0]

    def __getitem__(self, idx):
        return self.predictors[idx], self.predictands[idx]

class CNN(nn.Module):
    def __init__(self, num_input_time_steps=1, print_feature_dimension=False):
        """
        inputs
        -------
            num_input_time_steps        (int) : the number of input time
                                                steps in the predictor
            print_feature_dimension    (bool) : whether or not to print
                                                out the dimension of the features
                                                extracted from the conv layers
        """
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(num_input_time_steps, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.print_layer = Print()
        
        #ATTENTION EXERCISE 9: print out the dimension of the extracted features from 
        #the conv layers for setting the dimension of the linear layer!
        #Using the print_layer, we find that the dimensions are 
        #(batch_size, 16, 42, 87)
        self.fc1 = nn.Linear(16 * 12 * 21, 24)
        self.fc2 = nn.Linear(24, 20)
        self.fc3 = nn.Linear(20, 1)
        self.print_feature_dimension = print_feature_dimension

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        if self.print_feature_dimension:
          x = self.print_layer(x)
        x = x.view(-1, 16 * 12 * 21)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Print(nn.Module):
    """
    This class prints out the size of the features
    """
    def forward(self, x):
        print(x.size())
        return x

def train_network(net, criterion, optimizer, trainloader, testloader, 
                  experiment_name, num_epochs=50):
  """
  inputs
  ------

      net               (nn.Module)   : the neural network architecture
      criterion         (nn)          : the loss function (i.e. root mean squared error)
      optimizer         (torch.optim) : the optimizer to use update the neural network 
                                        architecture to minimize the loss function
      trainloader       (torch.utils.data.DataLoader): dataloader that loads the
                                        predictors and predictands
                                        for the train dataset
      testloader        (torch.utils.data. DataLoader): dataloader that loads the
                                        predictors and predictands
                                        for the test dataset
  outputs
  -------
      predictions (np.array), and saves the trained neural network as a .pt file
  """
  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  net = net.to(device)
  best_loss = np.infty
  train_losses, test_losses = [], []

  for epoch in range(num_epochs):
    for mode, data_loader in [('train', trainloader), ('test', testloader)]:
      #Set the model to train mode to allow its weights to be updated
      #while training
      if mode == 'train':
        net.train()

      #Set the model to eval model to prevent its weights from being updated
      #while testing
      elif mode == 'test':
        net.eval()

      running_loss = 0.0
      for i, data in enumerate(data_loader):
          # get a mini-batch of predictors and predictands
          batch_predictors, batch_predictands = data
          batch_predictands = batch_predictands.to(device)
          batch_predictors = batch_predictors.to(device)

          # zero the parameter gradients
          optimizer.zero_grad()

          #calculate the predictions of the current neural network
          predictions = net(batch_predictors).squeeze()

          #quantify the quality of the predictions using a
          #loss function (aka criterion) that is differentiable
          loss = criterion(predictions, batch_predictands)

          if mode == 'train':
            #the 'backward pass: calculates the gradients of each weight
            #of the neural network with respect to the loss
            loss.backward()

            #the optimizer updates the weights of the neural network
            #based on the gradients calculated above and the choice
            #of optimization algorithm
            optimizer.step()
          
          #Save the model weights that have the best performance!
        

          running_loss += loss.item()
      if running_loss < best_loss and mode == 'test':
          best_loss = running_loss
          torch.save(net, '{}.pt'.format(experiment_name))
      print('{} Set: Epoch {:02d}. loss: {:3f}'.format(mode, epoch+1, \
                                            running_loss/len(data_loader)))
      if mode == 'train':
          train_losses.append(running_loss/len(data_loader))
      else:
          test_losses.append(running_loss/len(data_loader))
    
  net = torch.load('{}.pt'.format(experiment_name))
  net.eval()
  net.to(device)
  
  #the remainder of this notebook calculates the predictions of the best
  #saved model
  predictions = np.asarray([])
  for i, data in enumerate(testloader):
    batch_predictors, batch_predictands = data
    batch_predictands = batch_predictands.to(device)
    batch_predictors = batch_predictors.to(device)

    batch_predictions = net(batch_predictors).squeeze()
    #Edge case: if there is 1 item in the batch, batch_predictions becomes a float
    #not a Tensor. the if statement below converts it to a Tensor
    #so that it is compatible with np.concatenate
    if len(batch_predictions.size()) == 0:
      batch_predictions = torch.Tensor([batch_predictions])
    predictions = np.concatenate([predictions, batch_predictions.detach().cpu().numpy()])
  return predictions, train_losses, test_losses

if run_flag == 1:
   plt.close('all')
# ================================================================
# Read data
# ================================================================
   ens_sel = 10
   time_lag = 0

# Load labeled data
   dirname = '/stormtrack/data4/share/deep_learning/data/'
#   filename = 'DampingTerm_NAtl_1920_2005_EnsAll.nc'
   filename = 'CESM1LE_SST_Anomaly_NAtl_1920_2005_EnsAll.nc'
   ds = xr.open_dataset(dirname + filename, decode_times=True)
#   var_tmp1 = ds['Damping_Term'][:,:,:,:].data.copy()
   var_tmp1 = ds['SST'][:,:,:,:].data.copy()
   lat = ds["lat"][:].data.copy()
   lon = ds["lon"][:].data.copy()
   ds.close()

   dirname = '/stormtrack/data4/share/deep_learning/data/'
   filename = 'DampingTerm_NAtl_1920_2005_EnsAll.nc'
   ds = xr.open_dataset(dirname + filename, decode_times=True)
   var_tmp3 = ds['Damping_Term'][:,:,:,:].data.copy()
   ds.close()

   ny = len(lat)
   nx = len(lon)
   ens_num = var_tmp1.shape[0]
   nt = var_tmp1.shape[1]

   mask1 = np.sum(np.sum(var_tmp1, axis=0), axis=0)
   mask1 = mask1/mask1

   dirname = '/stormtrack/data4/share/deep_learning/data/'
   filename = 'CESM1LE_SST_Anomaly_NAtl_1920_2005_EnsAll.nc'
   ds = xr.open_dataset(dirname + filename, decode_times=True)
   var_tmp2 = ds['SST'][:,:,:,:].data.copy()
   ds.close()

   mask2 = np.sum(np.sum(var_tmp2, axis=0), axis=0)
   mask2 = mask2/mask2
   mask_all = mask1*mask2

   var_tmp1[np.isnan(var_tmp1)] = 0.
   var1_comb = np.zeros((nt,time_lag+1,ny,nx))
   var_ts = np.zeros((nt))

   for NT in range(nt):
       var1_comb[NT,0,:,:] = var_tmp1[ens_sel,NT,:,:]
       var_ts[NT] = np.nanmean(var_tmp2[ens_sel,NT,:,:])

   var_tmp3[np.isnan(var_tmp3)] = 0.
   var3_comb = np.zeros((nt,time_lag+1,ny,nx))
   for NT in range(nt):
       var3_comb[NT,0,:,:] = var_tmp3[ens_sel,NT,:,:]

# ================================================================
# Train model
# ================================================================
   num_input_time_steps = 1
   train_predictors = var1_comb[0:800,:,:,:].astype(np.float32)
   train_predictands = var_ts[0:800].astype(np.float32)
   test_predictors = var1_comb[-100:,:,:,:].astype(np.float32)
   test_predictands = var_ts[-100:].astype(np.float32)
   test2_predictors = var3_comb[-100:,:,:,:].astype(np.float32)
   test2_predictands = var_ts[-100:].astype(np.float32)

#Convert the numpy ararys into AMV_dataset, which is a subset of the 
#torch.utils.data.Dataset class.  This class is compatible with
#the torch dataloader, which allows for data loading for a CNN
   train_dataset = AMV_dataset(train_predictors, train_predictands)
   test_dataset = AMV_dataset(test_predictors, test_predictands)
   test2_dataset = AMV_dataset(test2_predictors, test2_predictands)

#Create a torch.utils.data.DataLoader from the ENSODatasets() created earlier!
#the similarity between the name DataLoader and Dataset in the pytorch API is unfortunate...
   trainloader = DataLoader(train_dataset, batch_size=5)
   testloader = DataLoader(test_dataset, batch_size=5)
   net = CNN(num_input_time_steps=num_input_time_steps)

#   optimizer = optim.Adam(net.parameters(), lr=0.00001)
   optimizer = optim.SGD(net.parameters(), lr=0.01, weight_decay=0.001)

   experiment_name = "two_layer_CNN"
   predictions, train_losses, test_losses = train_network(net, nn.MSELoss(), 
                  optimizer, trainloader, testloader, experiment_name)

# Plot out loss function against epoch
   plt.figure()
   plt.plot(train_losses, label='Train Loss')
   plt.plot(test_losses, label='Test Loss')
   plt.xlabel('Epoch')
   plt.ylabel('Loss')
   plt.title('Performance of {} Neural Network During Training'.format(experiment_name))
   plt.legend(loc='best')

   corr, _ = pearsonr(test_predictands, predictions)
   rmse = mean_squared_error(test_predictands, predictions) ** 0.5

   print("corr = ", corr)
   print("rmse = ", rmse)

# Plot out predictand time series
   tt = np.linspace(1,len(predictions),len(predictions)) 
   plt.figure()
   plt.plot(tt,predictions,'r-',label='predicted AMV')
   plt.plot(tt,test_predictands,'b-',label='gound truth')
   plt.legend()

#   plot_nino_time_series(test_predictands, predictions, '{} Predictions. \n Corr: {:3f}. \n RMSE: {:3f}.'.format(experiment_name, corr, rmse))

   plt.figure()
   plt.contourf(lon,lat,var1_comb[111,:,:,:].squeeze(), levels=np.linspace(-2,2,21),cmap='RdBu_r')
   plt.colorbar()

   plt.show()


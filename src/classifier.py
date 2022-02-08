from xml.parsers.expat import model
from sklearn import datasets
import numpy as np
import torch
import torch.nn as nn 



from tqdm import tqdm
import os
import math
import pandas as pd
from collections import Counter

from torch.utils.tensorboard import SummaryWriter

## Base Class for classification
from configuration import config_deeplearing
from models import MLP, CNN


my_config = config_deeplearing()


## INPUTS
PATH            = my_config['PATH']['DATA_PATH']
TRAIN_FOLDER    = my_config['PATH']['TRAIN_FOLDER']
TEST_FOLDER     = my_config['PATH']['TEST_FOLDER']
CHECKPOINT_PATH = my_config['PATH']['CHECKPOINT_PATH']
TSBOARD_PATH    = my_config['PATH']['TSBOARD_PATH']
BATCH_SIZE      = int(my_config['PARAMETERS_DEEP']['BATCH_SIZE'])
VAL_SIZE_PROP   = float(my_config['PARAMETERS_DEEP']['VAL_SIZE_PROP'])
NUM_WORKER      = int(my_config['PARAMETERS_DEEP']['NUM_WORKER'])
LR              = float(my_config['PARAMETERS_DEEP']['LR'])
NUMBER_EPOCHS   = int(my_config['PARAMETERS_DEEP']['NUMBER_EPOCHS'])
CROP_SIZE       = int(my_config['PARAMETERS_DEEP']['CROP_SIZE'])
NUMBER_OUTPUTS  = int(my_config['PARAMETERS_DEEP']['NUMBER_OUTPUTS'])
PATIENCE        = int(my_config['PARAMETERS_DEEP']['PATIENCE'])
DELTA           = int(my_config['PARAMETERS_DEEP']['DELTA'])
UNFREEZE        = int(my_config['PARAMETERS_DEEP']['UNFREEZE'])
## GLOBALS
NUMBER_INPUTS   = 3 * CROP_SIZE * CROP_SIZE
TRAIN_PATH      = PATH + TRAIN_FOLDER
TEST_PATH       = PATH + TEST_FOLDER



class EarlyStopping:
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience : int =PATIENCE, delta: int =DELTA, path : str = CHECKPOINT_PATH + 'checkpoint_test.pt'):
        """[Constructor]

        Args:
            patience (int, optional): [value for patience]. Defaults to PATIENCE.
            delta (int, optional): [value for delta]. Defaults to DELTA.
            path (str, optional): [path to save chackpoints]. Defaults to CHECKPOINT_PATH+'checkpoint_test.pt'.
        """
        self.patience = patience
        self.delta = delta
        self.path= path
        self.counter = 0
        self.best_score = None
        self.early_stop = False


    def __call__(self, val_loss, model):
        """[Call]

        Args:
            val_loss ([float]): [Value of the loss]
            model ([type]): [model used]
        """
        if self.best_score is None:
            self.best_score = val_loss
            self.save_checkpoint(model)
        elif val_loss > self.best_score:
            self.counter +=1
            if self.counter >= self.patience:
                self.early_stop = True 
        else:
            self.best_score = val_loss
            self.save_checkpoint(model)# Save model
            # reset counter if validation loss improves
            self.counter = 0      

    def save_checkpoint(self, model):
        """[Save the best result of the pytoch model during the current training]

        Args:
            model ([type]): [pytorch model to save]
        """
        torch.save(model.state_dict(), self.path)

class Classifier():
    def __init__(self, data_dir : str, output_size : int, device, batch_size=BATCH_SIZE, lr=LR, stop_early=True):
        """[Constructor]

        Args:
            data_dir ([str]): [Path to the data]
            output_size ([int]): [Number of series of the problem]
            device ([type]): [Pytorch device (cpu or GPU)]
            batch_size ([int], optional): [Size of batch]. Defaults to BATCH_SIZE.
            lr ([type], optional): [Learning rate for the optimizer]. Defaults to LR.
            stop_early (bool, optional): [If you want to use early stopping : stop the training if there is no evolution after a given number of training]. Defaults to True.
        """
        self.data_dir = data_dir
        self.output_size = output_size
        self.device = device
        self.batch_size = batch_size
        self.lr = lr
        self.stop_early = stop_early
        self.name = ""
        
    def load_data(self):
        """[Load the dataset]

        Returns:
            [type]: [The train loader and the validation loader]
        """
        #TODO COMPLETE LOADDATA
        train_loader, val_loader = None

        return train_loader, val_loader
    
    
    def load_model(self, model_type='cnn'):
        """[Load a model (e.g. resnet50, cnn, mlp, p...)]

        Args:
            model_type (str, optional): [description]. Defaults to 'cnn'.
        """
        self.name = model_type
        # Load the right model from models.py
        if model_type == 'MLP':
            self.model = MLP()
        else: # default
            self.model = CNN()
            
        # Apply the model to the device
        self.model = self.model.to(self.device)
        # Apply Adam optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr) 

        self.criterion = nn.MSELoss()
            
    
            
    def fit_one_epoch(self, train_loader, epoch, num_epochs): 
        """[Method for fitting one epoch]

        Args:
            train_loader ([type]): [the train loader]
            epoch ([type]): [number of the current epoch]
            num_epochs ([type]): [total number of epochs]
        """

        # Lists for losses and accuracies
        train_losses = []
        # Turn model into train mode
        self.model.train()
        # Train...
        for i, (X, targets) in enumerate(tqdm(train_loader)):
            X = X.to(self.device)
            targets = targets.to(self.device)

            pred = self.model(X)
            loss = self.criterion(pred, targets)
            # loss.requires_grad = True
            loss.backward()
            self.optimizer.step()

            self.optimizer.zero_grad()

            train_losses.append(loss.item())


        train_loss = torch.tensor(train_losses).mean()    
        print(f'Epoch n° {epoch}/{num_epochs-1}')  
        print(f'Training Loss: {train_loss:.2f}')
        return train_loss
        
    def val_one_epoch(self, val_loader):
        """[Method for validating one epoch]

        Args:
            val_loader ([type]): [the validation loader]
        """
        # Lists for losses and accuracies
        val_losses = []
        # Turn model into eval mode
        self.model.eval()
        # Disabling gradient descent for evaluation
        with torch.no_grad():
            for (X, targets) in tqdm(val_loader):
                X = X.to(self.device)
                targets = targets.to(self.device)

                pred = self.model(X)
                loss = self.criterion(pred, targets)
                val_losses.append(loss.item())      

            self.val_loss = torch.tensor(val_losses).mean()
        
            print(f'Validation loss: {self.val_loss:.2f}')  
        return None
            
    def fit(self, train_loader, val_loader, num_epochs=NUMBER_EPOCHS, checkpoint_dir= CHECKPOINT_PATH):
        """[Method to fit a model : apply training et validation for a given number of epochs]

        Args:
            train_loader ([type]): [description]
            val_loader ([type]): [description]
            num_epochs ([int], optional): [number of epochs for fitting]. Defaults to NUMBER_EPOCHS.
            unfreeze_after ([int], optional): [parameter for unfreezing parameters of a pre-trained model when doing transfert learining]. Defaults to UNFREEZE.
            checkpoint_dir ([str], optional): [path to the chackpoint folder]. Defaults to CHECKPOINT_PATH.
        """
        
        # Start a tensorboard for monitoring
        if not os.path.exists(TSBOARD_PATH):
            # Create Tensorboard directory if not exists
            os.mkdir(TSBOARD_PATH)

        # Compute the number of the tensorboard exist from the kind of the model
        ltb             = os.listdir(TSBOARD_PATH)
        n               = 0
        path_to_tsboard = TSBOARD_PATH
        for tb in ltb:
            if tb.startswith(self.name):
                n += 1
        path_to_tsboard += self.name + "_" + str(n)
        os.mkdir(path_to_tsboard)   
        # Instanciate a summary writer
        TensorboardWriter   = SummaryWriter(log_dir = path_to_tsboard)
        
        
        # If one considers Early Stopping
        if self.stop_early:
            path = checkpoint_dir
            # Check if output path exits
            if not os.path.exists(path):
                print("Creating output path : ", path)
                os.makedirs(path)
            # Compute the number of the cp prediction from the kind of the model
            lcp    = os.listdir(path)
            n       = 0
            for cp in lcp:
                if cp.startswith(self.name):
                    n += 1
            path += self.name + "_" + str(n) + ".pt"        
            early_stopping = EarlyStopping(
                                            patience=PATIENCE,
                                            path = path)

        # Fit... for num_epochs epochs
        for epoch in range(num_epochs):
            for param in self.model.parameters():
                param.requires_grad = True
            train_loss  = self.fit_one_epoch(train_loader, epoch, num_epochs)
            self.val_one_epoch(val_loader)
            if self.stop_early:
                early_stopping(self.val_loss, self.model)
                if early_stopping.early_stop:
                    print('Early Stopping')
                    print(f'Best validation loss: {early_stopping.best_score}')
                    break
            # Add metrics to the tensorbaord 
            TensorboardWriter.add_scalar('eval loss', float(self.val_loss), epoch)
            TensorboardWriter.add_scalar('train loss', float(train_loss), epoch)

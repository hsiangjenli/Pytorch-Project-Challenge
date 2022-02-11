"""
@author: Hsiang-Jen Li
@email: hsiangjenli@gmail.com
"""

from _model.myModel import mAR
from _preprocessing.addFeatures import logReturn
from _preprocessing.TrainTest import TTS
from sklearn.metrics import mean_squared_error as MSE

import pandas as pd
import numpy as np

from itertools import product


import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataPATH = './Data/raw-TWWI-2011To2021.csv'

rawDF = pd.read_csv(dataPATH)

windowsParams = [10]
lrParams = [0.01]
epochParams = [200]

allProduct = product(windowsParams, lrParams, epochParams)


if __name__ == "__main__":

    for c in allProduct:
    
        windows, lr, epochs = c[0], c[1], c[2] 
        
        # Preprocessing the Data
        ## Log-Return and Rolling-window
        '''
        Transform TWII to Log-Return and using Rolling-window to add more Features.
        '''    
        LR = logReturn()
        LR.fit(df=rawDF['發行量加權股價指數'], windows=windows) 
        x, y = LR.rollingX, LR.rollingY.reshape(len(LR.rollingY), 1)
        x = np.nan_to_num(x, 0)
        model = mAR(input_size=x.shape[1], hidden_size=10)
         
        # Using GPU or CPU
        model.to(device)
        x = torch.Tensor(x)
        y = torch.Tensor(y)
        x = x.to(device)
        y = y.to(device)
        
        # Train-Test-Split
        tts = TTS()
        tts.fit(X=x, Y=y)
        train_X, train_Y = tts.train_X, tts.train_Y
            
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
            
        for epoch in range(epochs):
            out = model(train_X)
            optimizer.zero_grad()
                
            loss = criterion(out, train_Y)
            loss.backward()
                
            optimizer.step()
                
            torch.cuda.empty_cache()
                
            if epoch % 100 == 0:
                print (f'Epoch : {epoch:5d}\nLoss: {loss.item()}\n{"="*50}')
            
        y_pred_lr = model(x)
        y_pred_lr = y_pred_lr.cpu().detach().numpy().reshape(-1)
        y_pred = LR.backwardR(logR=y_pred_lr, pred=True)
        
        ts = len(tts.test_X)
        mse = MSE(y_true=rawDF['發行量加權股價指數'].values[-ts:], y_pred=y_pred[-ts:])
            
        print(mse)
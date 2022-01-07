import torch.nn as nn
import pandas as pd
import numpy as np
import datetime
import os

class mLSTM_Basic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(mLSTM_Basic, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size=input_size, 
                            hidden_size=hidden_size, 
                            num_layers=num_layers)
        
        self.fc = nn.Linear(in_features=hidden_size,
                            out_features=output_size)
        
    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        out = self.fc(out[:,-1])
        return out

def fit(
        
        model: str, modelDetails, criterion,
        optimizer: str, optimizerDetails, lr: int,
        addFeatures, windows: int,
        
        epochs: int, tmehods:str
        
        ):
    
    tTime = datetime.datetime.now()
    
    modelPATH = f'{model}-{optimizer}-{lr}-{tmehods}'


class Assistant:
    def __init__(self):
        pass
    
    def fit(self, 
            
            model: str, modelDetails, criterion,
            optimizer: str, optimizerDetails, lr: int,
            addFeatures, windows: int,
            epochs: int,
            loss: list, testLoss,          
            
            ):
        self.fitter()
        self.plotter()
        self.trainingReports()
        
    def fitter(self):
        
        
        pass
    def plotter(self):
        pass
    def trainingReports(self):
        pass


def trainingReports(
        
        tTime,
        model: str, modelDetails, criterion,
        optimizer: str, optimizerDetails, lr: int,
        addFeatures, windows: int,
        
        epochs: int,
        
        loss: list, testLoss,
        
        modelPATH:str

        ):
    
    eTime = datetime.datetime.now() - tTime    
    
    reports = {
        
        'Date': tTime,
        'Executive time': eTime,
        
        'Model': model,
        'Model-stat-dict': modelDetails,
        'Criterion': criterion,
        
        'Optimizer': optimizer,
        'Optimizer-stat-dict': optimizerDetails,
        'Learning-Rate': lr,
        
        'Add-Features': addFeatures,
        'Windows': windows,
        
        'Epochs': epochs,
        
        'Loss': loss,
        'min-Loss': min(loss),
        'min-Loss-occur': np.argmin(loss)*100,
        'test-Loss': testLoss,
        
        'Save-model-path': modelPATH
        }
    
    report = pd.DataFrame(reports, index=[0])
    
    csvPATH = 'Training Report.csv'
    
    if os.path.exists(csvPATH):
        report.to_csv(csvPATH, index=False, mode = 'a', header=False)
    else:
        report.to_csv(csvPATH, index=False)
    
    
    
    
    
    
    
    
    
    
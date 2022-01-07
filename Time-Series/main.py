# -*- coding: utf-8 -*-
"""
@author: Hsiang-Jen Li
@email: hsiangjenli@gmail.com
"""

from _model.myModel import *
from _preprocessing.addFeatures import logReturn

import pandas as pd
import copy

from itertools import product

dataPATH = './Data/raw-TWWI-2011To2021.csv'

rawDF = pd.read_csv(dataPATH)

windowsParams = [5, 10, 30]
lrParams = [0.01, 0.001]
epochParams = [1000, 2000]

output_size = 1

allProduct = product(windowsParams, lrParams, epochParams)


if __name__ == "__main__":
    
    for c in allProduct:
        # Set Params
        windows, lr, epochs = c[0], c[1], c[2] 
        
        # Preprocessing the Data
        ## Log-Return and Rolling-window
        '''
        Transform Index to Log-Return and using Rolling-window to add more Features.
        '''
        LR = logReturn()
        LR.fit(df=rawDF, windows=windows) 
        x, y = LR.rollingX, LR.rollingY
        
        
            
    pass

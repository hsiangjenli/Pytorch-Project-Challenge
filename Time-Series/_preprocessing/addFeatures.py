import numpy as np
import copy
import pandas as pd
import talib

nullRow = pd.Series({'null': 'null'})

class logReturn:
    def __init__(self):
        pass
    
    def fit(self, df, windows:int=0):
        self.df = copy.deepcopy(df)
        self.rdf = copy.deepcopy(df)
        self.windows = windows
        self.transformR()
        self.rollingWindow()

    def transformR(self):
        self.lr = np.log(self.df/self.df.shift(1))
        return self
    
    def backwardR(self, logR, pred:bool=False):
        if pred:
            ndf = copy.deepcopy(self.df)
            return np.exp(logR) * ndf[self.windows:].shift(1)
        else:
            return np.exp(logR) * self.df.shift(1)
    
    def rollingWindow(self):
        rolling = len(self.lr) - self.windows
        self.rollingX = np.array([self.lr[i: i+self.windows] for i in range(rolling)])
        self.rollingY = np.array([self.lr[i+self.windows] for i in range(rolling)])
        return self

class technicalA:
    def __init__(self):
        pass
    def fit(self, df):
        self.df = copy.deepcopy(df)
        self.setVariableName()
        self.generateMoreFeatures()
    def setVariableName(self):
        self.twiiIndex = self.df['發行量加權股價指數'].shift(1)
        self.volumePrice = self.df['成交金額'].shift(1)
        self.volumeEquity = self.df['成交股數'].shift(1)
        self.volume = self.df['成交筆數'].shift(1)
        return self
    def generateMoreFeatures(self):
        pass
import copy
class TTS:
    def __init__(self):
        pass
    def fit(self, X, Y, train_size=0.7):
        self.X = copy.deepcopy(X)
        self.Y = copy.deepcopy(Y)
        self.train_size = train_size
        self.size()
        self.trainSet()
        self.testSet()
    def size(self):
        self.s = len(self.Y)
    def trainSet(self):
        self.train_X = self.X[0: self.s]
        self.train_Y = self.Y[0: self.s]
    def testSet(self):
        self.test_X = self.X[self.s:]
        self.test_Y = self.Y[self.s:]
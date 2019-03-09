import numpy as np
import pandas as pd
import sklearn.neighbors as neg
import data_utils as ut

np.random.seed(777)

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict1 = pickle.load(fo, encoding='bytes')
    return dict1


Xtr, Ytr, Xte, Yte = ut.load_CIFAR10('e:/CS231n/data/')
Xtr_rows = Xtr.reshape(Xtr.shape[0], 32*32*3)
Xte_rows = Xte.reshape(Xte.shape[0], 32*32*3)

Xte_cv = Xte_rows[:5000]
Yte_cv = Yte[:5000]
Xte_test = Xte_rows[5000:]
Yte_test = Yte[5000:]

nn = neg.KNeighborsClassifier(n_neighbors=3, metric='manhattan', p=1)
nn.fit(Xtr_rows, Ytr)
yhat = nn.predict(Xte_cv)

print('acc: %f' % (np.mean(Yte_cv == yhat)))

class NearestNeighbor(object):
    def __init__(self):
        pass
    
    def train(self, X, y):
        ## lazy learner !! ##
        self.Xtr = X
        self.ytr = y
        
    def predict(self, X):
        num_test = X.shape[0]
        Y_hat = np.zeros(num_test, dtype=self.ytr.dtype)
        
        ## l1 (manhattan)
        for i in range(num_test):
            dist = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            l2_dist = np.sqrt(np.sum(np.square(self.Xtr - X[i, :]), axis=1))
            min_idx = np.argmin(dist)
            Y_hat[i] = self.Ytr[min_idx]
        
        return Y_hat
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
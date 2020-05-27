# -*- coding: utf-8 -*-
#!/usr/bin/env python
__author__ = "Andres Mendez-Vazquez"
__copyright__ = "Copyright 2019"
__credits__ = ["Andres Mendez-Vazquez"]
__license__ = "Closed"
__version__ = "v1.0.0"
__maintainer__ = "Andres Mendez-Vazquez"
__email =  "kajuna0kajuna@gmail.com"
__status__ = "Development"

import numpy as np
from normalization import Soft_Max
import math
from pandas_datareader import get_data_yahoo as gy

def FinancialData(symbol='AAPL', start='2015-01-01', end='2020-01-01'):
    X = gy(symbol, start, end)['Adj Close']

    return X

def DataGenerationFD(T_slen):
    
    Fd = FinancialData()
    
    data = Fd.values

    ndata, mean, std = Soft_Max(data)

    X = []
    Y = []
    
    num_records = len(ndata) - T_slen
        
    for i in range(num_records - T_slen):
        X.append(ndata[i:i+T_slen])
        Y.append(ndata[i+T_slen])
        
    X = np.array(X)
    X = np.expand_dims(X, axis=2)
    
    Y = np.array(Y)
    Y = np.expand_dims(Y, axis=1)
    
    X_val = []
    Y_val = []
    
    for i in range(num_records - T_slen, num_records):
        X_val.append(ndata[i:i+T_slen])
        Y_val.append(ndata[i+T_slen])
        
    X_val = np.array(X_val)
    X_val = np.expand_dims(X_val, axis=2)
    
    Y_val = np.array(Y_val)
    Y_val = np.expand_dims(Y_val, axis=1)
        
    return X, Y, X_val, Y_val, mean, std


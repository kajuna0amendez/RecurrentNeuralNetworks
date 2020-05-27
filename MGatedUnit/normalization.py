#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python
__author__ = "Andres Mendez-Vazquez"
__copyright__ = "Copyright 2019"
__credits__ = ["Andres Mendez-Vazquez"]
__license__ = "Open"
__version__ = "v1.0.0"
__maintainer__ = "Andres Mendez-Vazquez"
__email =  "kajuna0kajuna@gmail.com"
__status__ = "Development"

import numpy as np 


def Inv_Soft_Max(Y, mean, std):
    
    return mean-std*np.log((1-Y)/Y)

def Soft_Max(X):
    
    mean = np.mean(X)
    std = np.std(X)
    
    return 1.0 / (1.0 + np.exp((X-mean)/std)), mean, std
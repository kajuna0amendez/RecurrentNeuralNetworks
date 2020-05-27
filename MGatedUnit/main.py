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


from model import Run_train, EstimationMGU
from DataGeneration import DataGenerationFD
from normalization import Inv_Soft_Max 
import matplotlib.pyplot as plt
from config import *
 
if __name__ == '__main__':
    
    X, Y, X_val, Y_val, mean, std = DataGenerationFD(T_slen)
    
    W, U , V, D, bf, bz = Run_train(X, Y, X_val, Y_val, adam_alpha_config,\
                                    minbatch_alpha, Beta1_config, Beta2_config,\
                                    epsilon_adam, minbatch, num_epochs)

    y_est = EstimationMGU(W, U , V, D, bf, bz, X_val, Y_val)
    Y_train = EstimationMGU(W, U , V, D, bf, bz, X, Y)
    
    print(y_est.shape)
    
    Real_Y_val = Inv_Soft_Max(Y_val, mean, std)
    Real_Y_est = Inv_Soft_Max(y_est, mean, std)
    
    plt.figure()
    plt.plot(y_est, 'g', label = 'Estimate Y Validation')
    plt.plot(Y_val, 'r', label = 'Validation Y')
    plt.legend()
    plt.figure()
    plt.plot(Y_train, 'g', label = 'Estimate Train Y')
    plt.plot(Y, 'r', label = 'Train Y')
    plt.legend()
    
    plt.figure()
    plt.plot(Real_Y_est, 'g', label = 'Estimate Train Y')
    plt.plot(Real_Y_val, 'r', label = 'Validation Y')
    plt.legend()
    plt.show()
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
import time
from config import *
import numba
from numba import prange

"""
The Implementation of a Minumum Gated Unit to avoid the RNN backprop using
the concept of understanding your problem in automatic differentiation
+ https://arxiv.org/abs/1603.09420
"""

@numba.jit(nopython=True, parallel = False)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

@numba.jit(nopython=True, parallel = False)
def dsigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

@numba.jit(nopython=True, parallel = False)
def atanh(x):
    return np.tanh(x)

@numba.jit(nopython=True, parallel = False)
def adtanh(x):
    return 1-np.tanh(x)**2

@numba.jit(nopython=True, parallel = False)
def Init_MGU_weights(Dampening = 0.1):
    """
    The Generation of the MGU weights
    """
    W = np.random.uniform(0, Dampening, (output_dim, T_slen))
    
    U = np.random.uniform(0, Dampening, (output_dim, output_dim))
    
    bf = np.random.uniform(0, Dampening, (output_dim, 1))
    
    V = np.random.uniform(0, Dampening, (output_dim, T_slen))
    
    D = np.random.uniform(0, Dampening, (output_dim, output_dim))
    
    bz = np.random.uniform(0, Dampening, (output_dim, 1))
    
    return W, U , V, D, bf, bz

@numba.jit(nopython=True, parallel = True)
def LossMGU(W, U , V, D, bf, bz, X, Y):
    """
    Loss function for the forward pass
    """
    # check loss on train
    loss = 0.0
    
    prev_h = np.zeros((output_dim, 1))
    
    for i in prange(Y.shape[0]):
        # Get samples
        x_s, y_s = X[i], Y[i]
        
        new_input = x_s
        mulW = np.dot(W, new_input) 
        mulU = np.dot(U, prev_h)
        f_t  = sigmoid(mulW + mulU + bf)
        mulV = np.dot(V, new_input)
        mulD = np.dot(D, f_t*prev_h)
        z_t = atanh( mulV + mulD + bz)
        h_t  =  f_t*prev_h + (1.0-f_t)*z_t
        
        prev_h = h_t
        

        # calculate error 
        loss_per_record = (y_s - f_t)**2 / 2
        loss += loss_per_record[0,0]

    return loss

@numba.jit(nopython=True, parallel = True)
def EstimationMGU(W, U , V, D, bf, bz, X, Y):
    """
    Estimation of the values
    """
    preds = np.zeros(Y.shape[0])
    
    prev_h = np.zeros((output_dim, 1))
    
    for i in range(Y.shape[0]):
        # Get samples
        x_s, y_s = X[i], Y[i]
        
        new_input = x_s
        mulW = np.dot(W, new_input) 
        mulU = np.dot(U, prev_h)
        f_t  = sigmoid(mulW + mulU + bf)
        mulV = np.dot(V, new_input)
        mulD = np.dot(D, f_t*prev_h)
        z_t = atanh( mulV + mulD + bz)
        h_t  =  f_t*prev_h + (1.0-f_t)*z_t
        
        prev_h = h_t  
        
        preds[i] = f_t[0,0]
    
    return preds

@numba.jit(nopython=True, parallel = False)
def Train_MGU_ADA_SigOL(W, U , V, D, bf, bz, X, Y, X_val, Y_val, aalpha,\
                        malpha, Beta1, Beta2, epsilon, mbatch , nepochs):
    """
    The training of the recurrent network
    """
    W_m_t = np.zeros(W.shape)
    W_v_t = np.zeros(W.shape)
    U_m_t = np.zeros(U.shape)
    U_v_t = np.zeros(U.shape)
    V_m_t = np.zeros(V.shape)
    V_v_t = np.zeros(V.shape)
    D_m_t = np.zeros(D.shape)
    D_v_t = np.zeros(D.shape)
    bf_m_t = np.zeros(bf.shape)
    bf_v_t = np.zeros(bf.shape)
    bz_m_t = np.zeros(bz.shape)
    bz_v_t = np.zeros(bz.shape)
    #millis = 0.0
    
    # Random seed for init 
    np.random.seed(millis)
    time_ada = 1

    
    for epoch in range(nepochs):
        # Reset to zero state at the begining of the epoch
        prev_h = np.zeros((output_dim, 1))
        prev_f = np.zeros((output_dim, 1))
        # Loss Train
        Loss_Train = LossMGU(W, U , V, D, bf, bz, X, Y)
        Loss_Val   = LossMGU(W, U , V, D, bf, bz, X_val, Y_val)
        
        if epoch%iter_value == 0:
            print('Total # of Epochs:', nepochs )
            print('Epoch: ', epoch + 1, ', Train Loss: ', Loss_Train,\
                  ', Val Loss: ', Loss_Val)
            print('Learning Rate: ', aalpha)
      
        for kb in range(Y.shape[0]//minbatch-1):
            batch = np.arange(kb*mbatch, kb*mbatch+mbatch+1)
            
            # Derivative no time
            dW  = np.zeros(W.shape)
            dU  = np.zeros(U.shape)
            dbf = np.zeros(bf.shape)
            dV = np.zeros(V.shape)
            dD  = np.zeros(D.shape)
            dbz = np.zeros(bz.shape)
            
        
            x_prev = np.zeros(X[0].shape)
            
            for cnt, i in enumerate(batch):
                # Get samples
                x_s, y_s = X[i], Y[i]
                new_input = x_s
                
########################## Forward on the Derivatives #########################                
                mulW = np.dot(W, new_input) 
                mulU = np.dot(U, prev_h)
                f_t  = sigmoid(mulW + mulU + bf)
                #print(ft.shape)
                mulV = np.dot(V, new_input)
                mulD = np.dot(U, f_t*prev_h)
                z_t = atanh( mulV + mulD + bz)
                h_t  =  f_t*prev_h + (1.0-f_t)*z_t
                # derivative of error at ouput
                dL = (f_t - y_s)
            
###############################################################################    
        
                #Backpropagation Pass Upper Layer
                ddL   = dL*dsigmoid(mulW + mulU + bf)
                dW_t  = np.dot(ddL , x_s.T)
                dU_t  = np.dot(ddL , prev_h.T)
                dbf_t = np.dot(ddL , np.eye(output_dim))

                # Second Layer
                dft_ht = np.dot(dsigmoid(mulW + mulU + bf), U)
                dzt_V = adtanh(mulV + mulD + bz)*x_prev.T
                dV_t = dL*dft_ht*(1.0 - prev_f)*dzt_V 
                dD_t = dL*dft_ht*adtanh(mulV + mulD + bz)*f_t*prev_h
                dbz_t = dL*dft_ht*adtanh(mulV + mulD + bz)*1
    
                # Delay state
                prev_h = h_t
                prev_f = f_t
                
                # Generate the Deltas
                dW += malpha*dW_t 
                dU += malpha*dU_t
                dbf += malpha*dbf_t
                
                # Deltas for the second layer
                dV += malpha*dV_t
                dD += malpha*dD_t
                dbz += malpha*dbz_t 
    
            # update
            W_gt  = np.copy(dW) 
            U_gt  = np.copy(dU)
            bf_gt = np.copy(dbf)
            V_gt = np.copy(dV)
            D_gt = np.copy(dD)
            bz_gt = np.copy(dbz)


######################### ADAM ###############################################

            # Update biased first moment estimate
            W_m_t = Beta1*W_m_t + (1-Beta1)*W_gt
            U_m_t = Beta1*U_m_t + (1-Beta1)*U_gt
            bf_m_t = Beta1*bf_m_t + (1-Beta1)*bf_gt
            
            V_m_t = Beta1*V_m_t + (1-Beta1)*V_gt
            D_m_t = Beta1*D_m_t + (1-Beta1)*D_gt
            bz_m_t = Beta1*bz_m_t + (1-Beta1)*bz_gt
            

            # Update biased second raw moment estimate
            W_v_t = Beta2*W_v_t + (1-Beta2)*(W_gt**2)
            U_v_t = Beta2*U_v_t + (1-Beta2)*(U_gt**2)
            bf_v_t = Beta2*bf_v_t + (1-Beta2)*(bf_gt**2)
            
            V_v_t = Beta2*V_v_t + (1-Beta2)*(V_gt**2)
            D_v_t = Beta2*D_v_t + (1-Beta2)*(D_gt**2)
            bz_v_t = Beta2*bz_v_t + (1-Beta2)*(bz_gt**2)
            
            # Compute bias-corrected first moment estimate
            W_m_hat = W_m_t/(1-(Beta1**time_ada))
            U_m_hat = U_m_t/(1-(Beta1**time_ada))
            bf_m_hat = bf_m_t/(1-(Beta1**time_ada))
            
            V_m_hat = V_m_t/(1-(Beta1**time_ada))
            D_m_hat = D_m_t/(1-(Beta1**time_ada))
            bz_m_hat = bz_m_t/(1-(Beta1**time_ada))
            
            # Compute bias-corrected second moment estimate
            W_v_hat = W_v_t/(1-(Beta2**time_ada))
            U_v_hat = U_v_t/(1-(Beta2**time_ada))
            bf_v_hat = bf_v_t/(1-(Beta2**time_ada))
            
            V_v_hat = V_v_t/(1-(Beta2**time_ada))
            D_v_hat = D_v_t/(1-(Beta2**time_ada))
            bz_v_hat = bz_v_t/(1-(Beta2**time_ada))
            
######################### Gradient Updates ####################################
            
            alpha_t = aalpha*(np.sqrt(1-Beta2**time_ada)/(1-Beta1**time_ada))
            
            W -= alpha_t*(W_m_hat/((W_v_hat**0.5)+epsilon))
            U -= alpha_t*(U_m_hat/((U_v_hat**0.5)+epsilon))
            bf -= alpha_t*(bf_m_hat/((bf_v_hat**0.5)+epsilon))
            
            V -= alpha_t*(V_m_hat/((V_v_hat**0.5)+epsilon))
            D -= alpha_t*(D_m_hat/((D_v_hat**0.5)+epsilon))
            bz -= alpha_t*(bz_m_hat/((bz_v_hat**0.5)+epsilon))
    
            # Increase Time Ada
            time_ada +=1.0
            
    return W, U , V, D, bf, bz   

def Run_train(X, Y, X_val, Y_val, adam_alpha_config, alpha_minbatch, Beta1_config,\
              Beta2_config, epsilon_adam, minbatch, num_epochs):
    """
    Train System for MGU
    """
    W, U , V, D, bf, bz = Init_MGU_weights()
    
    return Train_MGU_ADA_SigOL(W, U , V, D, bf, bz, X, Y, X_val, Y_val, adam_alpha_config,\
                               alpha_minbatch, Beta1_config,Beta2_config,\
                               epsilon_adam, minbatch, num_epochs)         
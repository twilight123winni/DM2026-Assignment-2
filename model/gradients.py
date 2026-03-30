import numpy as np
from model.utils import onehot_array

def MSE_grad(y,y_pred):
	'''
	Derivative of MSE loss w.r.t y_pred (not w)
	'''
	return (2/len(y))*(y_pred-y)

def MAE_grad(y, y_pred):
    '''
    Derivative of MAE loss w.r.t y_pred
    公式: (1/n) * sign(y_pred - y)
    '''
    return np.sign(y_pred - y) / len(y)  # np.sign 會根據正負值回傳 1, -1 或 0

def logloss_sigmoid_grad(y,y_pred):
	'''
	Derivative of sigmoid + log loss combination is equivalent to derivative of MSE 
	'''
	return MSE_grad(y,y_pred)/2
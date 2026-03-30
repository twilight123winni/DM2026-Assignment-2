import numpy as np
from model.utils import get_train_val,plot_learning_curve

def initialize_weight(dim):
	W0 = np.array([[0]]) # bias, 1x1
	W= np.random.rand(dim,1)
	return np.concatenate((W0,W))

class LinearModel():
	def __init__(self,dim,is_reg,loss_fn,grad_fn,act_fn = lambda x: x):
		self.dim,self.act_fn,self.loss_fn,self.grad_fn,self.is_reg = dim,act_fn,loss_fn,grad_fn,is_reg

		self.W = initialize_weight(self.dim)
		self.train_losses=[]
		self.val_losses=[]

	def _ensure_bias_column(self, X):
		if X.shape[1] == self.dim: # 特徵數=實際維度，表示缺bias column
			X0 = np.array([[1] * X.shape[0]]).T #造出一整排高度跟數據一樣長、內容全是 1 的垂直直條
			return np.concatenate((X0, X), axis=1) #連結到X左方
		if X.shape[1] == self.dim + 1: #特徵數=實際維度+1，不用補bias column
			return X
		raise ValueError(f'Input feature dimension mismatch: expected {self.dim} (without bias) or {self.dim + 1} (with bias), got {X.shape[1]}') #如果兩種狀況都不是
	
	def fit(self,X,y,lr,reg_type='',reg_lambda=0,n_iteration=50,val_ratio=.2):
		'''
		Fit data using gradient descent and l1/l2 regularization
		'''
		X = self._ensure_bias_column(X)
		X_train,y_train,X_val,y_val = get_train_val(X,y,val_ratio)
		for i in range(n_iteration):      
			y_pred = self.act_fn(np.squeeze(X_train @ self.W)) # @ 矩陣乘法 #np.squeeze(...)：把多餘的維度拿掉，確保它是一個簡單的一維陣列。 # .act_fn:把「無限大到無限小」的計算結果，壓縮到一個特定的範圍內。
			# MSE loss for regression
			loss = self.loss_fn(y_train,y_pred)
			grad = self.grad_fn(y_train,y_pred) # shape (n,)
			grad_w = X_train.T @ grad # shape (dim,) # 將誤差反推回每個權重 W 身上
			
			if len(grad_w.shape)==1: grad_w = grad_w[:,None] # turn (dim,) to (dim,1)
			#ignore update of grad_w0 (bias term) since w0 does not contribute to regularization process
			
			if reg_type == 'l2': #處罰「太大的權重」
				grad_w+= 2*(reg_lambda/len(X_train))*self.W # (2 *lambda / m)* weight
				pass

			self.W-= lr*grad_w #更新權重
			#save training loss
			self.train_losses.append(loss)

			#predict validation set
			y_pred = self.act_fn(np.squeeze(X_val @ self.W))
			val_loss = self.loss_fn(y_val,y_pred)
			self.val_losses.append(val_loss)
			if (i+1) % 50 == 0:
				print(f'{i+1}. Training loss: {loss}, Val loss:{val_loss}')

		plot_learning_curve(self.train_losses,self.val_losses)

	def get_weight(self):
		return self.W
	def predict(self,X,thres=0.5):
		if X.shape[1] == self.dim:
			X0 = np.array([[1]*X.shape[0]]).T # nx1
			X = np.concatenate((X0,X),axis=1)
		y_pred= self.act_fn(np.squeeze(X @ self.W))
		if not self.is_reg:
			y_pred = (y_pred >= thres).astype(np.uint8)
		return y_pred
	def predict_proba(self,X):
		if self.is_reg:
			raise Exception('Cannot predict probability for regression')
		if X.shape[1] == self.dim:
			X0 = np.array([[1]*X.shape[0]]).T # nx1
			X = np.concatenate((X0,X),axis=1)
		return self.act_fn(np.squeeze(X @ self.W))
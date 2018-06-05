# Linear Regression
import numpy as np
from numpy.linalg import inv
import pickle
import numpy as np
import math

class Linear_Regression_Classifier(object):

  def __init__(self, X, y):
      self.features = X
      self.target = y

  def train(self,l):
      """STUDNET_CODE"""
      x = self.features
      y = self.target
      
      I = np.identity(len(x[1,:]))
      I[0,0] = 0

      xT = np.matrix.transpose(x)
      xTx = np.dot(xT,x) + l*I
      XTX = inv(xTx)
      XTX_xT = np.dot(XTX,xT) 
      
      W = XTX_xT.dot(y)

      W = W[np.newaxis]
      Y = np.dot(x,W.T)
      
      self.W = W
      
  def predict(self, X):
      wT = self.W.T
      y = X.dot(wT)
      return y
  
  def encoding_scalar_to_group(self,y):
      group_pred = np.around(y)
      group_pred = np.squeeze(group_pred).T
      return group_pred

  def encdoing_vector_to_group(self,y):
      group_pred = np.squeeze(y)
      group = [np.argmax(r,axis=0) for r in group_pred]
      return group

  def classifier_onehot(self,X):
      y = self.predict(X)
      group_pred=self.encdoing_vector_to_group(y)
      return group_pred
 
  def classifier_scalar(self,X):
      y = self.predict(X)
      group_pred=self.encoding_scalar_to_group(y)
      return group_pred
    

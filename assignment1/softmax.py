import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - W: C x D array of weights
  - X: D x N array of data. Data are D-dimensional columns
  - y: 1-dimensional array of length N with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to weights W, an array of same size as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  X = X.T
  W = W.T
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
      f = X[[i], :].dot(W)
      f -= np.max(f)
      denom = np.sum(np.exp(f))
      nom = np.exp(f[:,y[i]])
      loss += -np.log(nom / denom)
      for j in range(num_classes):
          if j == y[i]:
              dW[:, [y[i]]] += (np.exp(f[:, j]) / denom - 1) * X[[i], :].T
          else:
              dW[:, [j]] += (np.exp(f[:, j]) / denom) * X[[i], :].T
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW /= num_train
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  
  return loss, dW.T


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  X = X.T
  W = W.T
  dW = np.zeros_like(W)
  m = X.shape[0]

  
  # #############################################################################
  # # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # # Store the loss in loss and the gradient in dW. If you are not careful     #
  # # here, it is easy to run into numeric instability. Don't forget the        #
  # # regularization!                                                           #
  # #############################################################################
  classes = np.max(y, axis=0) + 1
  
  y_one_hot = np.eye(classes)[y.reshape(-1)] # one-hot vector
  
  score = np.matmul(X, W)
  
  score -= np.max(score, axis=1, keepdims=True)
  
  softmax = np.exp(score) / np.sum(np.exp(score), axis=1, keepdims=True)
  
  loss = -1./ m * np.sum(y_one_hot * np.log(softmax)) + 0.5 * reg * np.sum(W[1:,:]*W[1:,:])

  dW = np.dot(X.T, softmax - y_one_hot) / m
  
  dW += reg * W
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW.T

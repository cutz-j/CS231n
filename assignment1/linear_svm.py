import numpy as np
from random import shuffle
# from past.builtins import xrange
import sys

if sys.version_info >= (3, 0):
    def xrange(*args, **kwargs):
        return iter(range(*args, **kwargs))


def svm_loss_naive(W, X, y, reg):
    """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]
    num_class = W.shape[1]
    loss = 0.
    
    score = np.matmul(X, W) # shape (N, C)
    
    for i in range(num_train):
        for j in range(num_class):
            if j != y[i]:
                margin = score[i, j] - score[i, y[i]] + 1.
                if margin > 0:
                    loss += margin
                    dW[:,j] += X[i]
                
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # Add regularization to the gradient
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero
    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    classes = np.max(y) + 1
    # s: A numpy array of shape (N, C) containing scores
    s = np.dot(X, W) # shape(500, 10)
    y_one_hot = np.eye(classes)[y.reshape(-1)] # shape(500, 10)
    y_zero_hot = np.abs(y_one_hot - 1)
    s_one = np.sum(s * y_one_hot, axis=1)
    
    loss_mat = s - s_one.reshape(-1,1) + 1.
    loss_mat *= y_zero_hot
    loss_mat[loss_mat < 0] = 0
    loss_sum = np.sum(loss_mat)
    loss = loss_sum / num_train
    loss += reg * np.sum(W * W)


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################


    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    loss_mat[loss_mat > 0] = 1
    dW = np.dot(X.T, loss_mat)
    dW /= num_train
    dW += 2 * reg * W


    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return loss, dW

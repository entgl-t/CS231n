import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  summa_prob = 0
  scores = X.dot(W)
  num_items = len(X)
  num_classes = scores.shape[1]
  for i in range(num_items):
     prob = np.exp(scores[i,:])
     prob_norm = prob/np.sum(prob)
     loss += -np.log(prob_norm[y[i]])
     for j in range(num_classes):
         if j == y[i]:
             dW[:,j] += (prob_norm[y[i]] - 1)*X[i]
         else:
             dW[:,j] += prob_norm[j]*X[i]
     
  loss /= num_items
  loss += reg * np.sum(W*W) 
  dW /= num_items
  dW += reg*W    
        
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  num_items = X.shape[0]
  num_classes = scores.shape[1]
  prob = np.exp(scores)
  prob_norm = prob/np.sum(prob, axis=1,keepdims=True)


  correct_classes =  prob_norm[np.arange(prob_norm.shape[0]),y]
  loss = np.sum(-np.log(correct_classes))
  loss /= num_items
  loss += 0.5*reg * np.sum(W*W)


  temp = prob_norm
  temp[np.arange(temp.shape[0]),y] -= 1
  dW = X.T.dot(temp)
  dW /= num_items
  dW += reg*W  
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


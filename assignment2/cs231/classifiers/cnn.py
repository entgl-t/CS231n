from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.
        
        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        depth = input_dim[0]
        conv_dim = (num_filters, depth, filter_size, filter_size)
        H = input_dim[1]
        W = input_dim[2]
        HH = conv_dim[2]
        WW = conv_dim[3]
        p = (filter_size - 1) // 2
        stride = 1
        Hf = int((H + 2 * p - HH) / stride + 1)
        Wf = int((W + 2 * p - WW) / stride + 1)
        #1 + (H + 2 * pad - HH) / stride
        
        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        print(input_dim, conv_dim)
        self.params['W1'] = weight_scale * np.random.randn(*conv_dim)
        self.params['b1'] = np.zeros(conv_dim[0])
        self.params['W2'] = weight_scale * np.random.randn((Hf//2) * (Wf//2) * num_filters , hidden_dim)
        self.params['b2'] = np.zeros(hidden_dim)
        self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b3'] = np.zeros(num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        # Unpack variables from the params dictionary
        reg = self.reg
        
       
        scor_conv_layer, conv_params =  conv_relu_pool_forward(X, W1, b1, conv_param, pool_param) 
        #print('COnvShape',scor_conv_layer.shape)
        scor_hid_layer, hid_fc_params = affine_relu_forward(scor_conv_layer, W2, b2)
        #print('HidShape',scor_hid_layer.shape)
        scores, last_params = affine_forward(scor_hid_layer, W3, b3)
        #print('LastShape',scores.shape)
        
        num_train = X.shape[0]
        exp_scores = np.exp(scores)
        
        softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        

        all_net_params = (conv_params, hid_fc_params, last_params)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        correct_scores_norm = softmax[np.arange(softmax.shape[0]),y]
        loss = np.sum(-np.log(correct_scores_norm))
        loss /=  num_train
    
    
        loss +=0.5*reg*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3)) 

        # Computing gradients
        grads['W1'] = np.zeros(W1.shape)
        grads['b1'] = np.zeros(b1.shape)
        grads['W2'] = np.zeros(W2.shape)
        grads['b2'] = np.zeros(b2.shape)
        grads['W3'] = np.zeros(W3.shape)
        grads['b3'] = np.zeros(b3.shape)


        dL = softmax                                        						#(NxC)
        dL[np.arange(dL.shape[0]),y] -= 1     
        dL /= num_train 
        
        last_layer_derivatives = affine_backward(dL, last_params)               			#Scores layer derivatives
        grads['W3'] = last_layer_derivatives[1]
        #grads['W2'] = np.dot(scor_hidden_layer.T, dL)      
        grads['b3'] = last_layer_derivatives[2] 
        #print('GradW3',grads['W3'].shape, 'Gradb3', grads['b3'].shape)         
                                                                   
       
        dx_last_layer = last_layer_derivatives[0]         
       
        second_layer_derivatives = affine_relu_backward(dx_last_layer, hid_fc_params) 			# Affine layer derivatives
       
        grads['W2'] = second_layer_derivatives[1]
        grads['b2'] = second_layer_derivatives[2] 

        dx_second_layer = second_layer_derivatives[0] 
        #print('GradW2',grads['W2'].shape, 'Gradb2', grads['b2'].shape)

        first_layer_derivatives = conv_relu_pool_backward(dx_second_layer, conv_params) 	# Convolution layer derivatives

        grads['W1'] = first_layer_derivatives[1]          
        grads['b1'] = first_layer_derivatives[2] 

        dx_first_layer = first_layer_derivatives[0]
        #print('GradW1',grads['W1'].shape, 'Gradb1', grads['b1'].shape)
        

         
        grads['W3'] += reg * W3
        grads['W2'] += reg * W2
        grads['W1'] += reg * W1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

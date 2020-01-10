from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)
        #self.params['reg'] = reg

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        reg = self.reg
        
       
        scor_hidden_layer, start_params = affine_relu_forward(X,W1,b1)
        scores, hidL_params = affine_forward(scor_hidden_layer, W2, b2)
        all_net_params = (start_params, hidL_params)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        #Computing the loss
        num_train = X.shape[0]
        exp_scores = np.exp(scores)
        prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        correct_scores_norm = prob[np.arange(prob.shape[0]),y]
        loss = np.sum(-np.log(correct_scores_norm))
        loss /=  num_train
    
    
        loss +=0.5*reg*(np.sum(W1*W1) + np.sum(W2*W2))
        #print('===================', loss)

        # Computing gradients
        grads['W1'] = np.zeros(W1.shape)
        grads['b1'] = np.zeros(b1.shape)
        grads['W2'] = np.zeros(W2.shape)
        grads['b2'] = np.zeros(b2.shape)

        dL = prob                                         #(NxC)
        dL[np.arange(dL.shape[0]),y] -= 1     
        dL /= num_train 
        
        second_layer_derivatives = affine_backward(dL, hidL_params)
        grads['W2'] = second_layer_derivatives[1]
        #grads['W2'] = np.dot(scor_hidden_layer.T, dL)    # dL/dW2 size (HxC)  
        grads['b2'] = second_layer_derivatives[2]         #dL/db2  size (Cx1)

                                                           
   
        dL_dHidden = second_layer_derivatives[0]          #(NxH)
     
        first_layer_derivatives = affine_relu_backward(dL_dHidden, start_params)
        grads['W1'] = first_layer_derivatives[1]
        #grads['W1'] = np.dot(X.T , dL_dHidden)           #(DxH)
        grads['b1'] = first_layer_derivatives[2] 

        grads['W2'] += reg * W2
        grads['W1'] += reg * W1
     
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        #print('Num of all layers', self.num_layers)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        
        
        for num_of_layer in range(0 , self.num_layers):
                if(num_of_layer == 0):
                    self.params['W'+ str(num_of_layer + 1)] = weight_scale * np.random.randn(input_dim, hidden_dims[num_of_layer])
                    self.params['b'+ str(num_of_layer + 1)] = np.zeros(hidden_dims[num_of_layer])
                    if self.use_batchnorm:
                        self.params['gamma' + str(num_of_layer + 1)] = np.ones([1, hidden_dims[num_of_layer]])
                        self.params['beta' + str(num_of_layer + 1)] = np.zeros([1, hidden_dims[num_of_layer]])
                        #print('Hier  gamma' + str(num_of_layer + 1))
                    ''' 
                    if self.use_dropout:
                        self.params['mask' + str(num_of_layer + 1)] = (np.random.rand(hidden_dims[num_of_layer]) < dropout) / dropout
                    '''

                    
                    #print('First '+ str( self.params['W'+ str(num_of_layer + 1)].shape))

                elif(num_of_layer == self.num_layers - 1 ):
                      self.params['W' + str(num_of_layer + 1)] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
                      self.params['b' + str(num_of_layer + 1)] = np.zeros(num_classes)
                      '''
                      if self.use_batchnorm:
                           self.params['gamma' + str(num_of_layer + 1)] = np.ones([1, hidden_dims[-1]])
                           self.params['beta' + str(num_of_layer + 1)] = np.zeros([1, hidden_dims[-1]])
                           print('Hier last gamma' + str(num_of_layer + 1))
                      print('Last' + str( self.params['W'+ str(num_of_layer + 1)].shape))
                      '''
                else:
                      #print('PreMiddle' + str(num_of_layer))
                      self.params['W'+ str(num_of_layer + 1)] = weight_scale * np.random.randn( 
	   										hidden_dims[num_of_layer - 1]
										        ,hidden_dims[num_of_layer]
											)
                      self.params['b' + str(num_of_layer + 1)] = np.zeros(hidden_dims[num_of_layer])
                      if self.use_batchnorm:
                           self.params['gamma' + str(num_of_layer + 1)] = np.ones([1, hidden_dims[num_of_layer]])
                           self.params['beta' + str(num_of_layer + 1)] = np.zeros([1, hidden_dims[num_of_layer]])
                           #print('Hier gamma' + str(num_of_layer + 1))
                      '''
                      if self.use_dropout:
                           print('Size: ', [input_dim, hidden_dims[num_of_layer]])
                           self.params['mask' + str(num_of_layer + 1)] = (np.random.rand(hidden_dims[num_of_layer]) < dropout) / dropout
                      '''
                      
                      #print('Middle' + str( self.params['W'+ str(num_of_layer + 1)].shape))

                #print('W'+ str(num_of_layer + 1))
                
                
    
        #self.params['W' + str(num_of_layer + 2)] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
        #self.params['b' + str(num_of_layer + 2)] = np.zeros(num_classes)
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        
        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)
        
        
    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
                
        reg = self.reg
        first_params = None
        scor_Hidlayer = None
       

        # First iteration of forward path(Will be improved)

        if self.use_batchnorm or self.use_dropout :
                

                #With batch normalization 
                if self.use_batchnorm and (not self.use_dropout): 
                      #print('Batchnorm true and Dropout false')
                      scor_Hidlayer, first_params = self.affine_norm_relu_forward(X
										 ,self.params['W1']
										 ,self.params['b1']
	        								 ,self.params['gamma1']
										 ,self.params['beta1']
										 ,self.bn_params[0]                                    #{'mode':'train'}
										)
                      #print('=============Hidden================',scor_Hidlayer) 
                if self.use_batchnorm and self.use_dropout: 
                      scor_Hidlayer, first_params = self.affine_norm_relu_drop_forward(X
								                 ,self.params['W1']
								                 ,self.params['b1']
										 ,self.params['gamma1']
										 ,self.params['beta1']
										 ,self.bn_params[0]       
										 ,self.dropout_param
										 ) 
                      
                    

                      

                if (not self.use_batchnorm) and  self.use_dropout: 
                      print('Batchnorm false and Dropout true')
                     
                      scor_Hidlayer, first_params = self.affine_relu_drop_forward(X
								                  ,self.params['W1']
								                  ,self.params['b1']   
										  ,self.dropout_param
  										 )
                      
                     
        if (not self.use_batchnorm) and (not self.use_dropout):
                      scor_Hidlayer, first_params = affine_relu_forward(X
						                        ,self.params['W1']
						                        ,self.params['b1']
								      )
                      #print('Shape of:', scor_Hidlayer.shape)
        	
        scores = None
        last_params = None                                        
        scor_nextHidden = None
        feed_params = {}
        feed_params[0] = first_params

       
        
        for num_of_layer in range(1,self.num_layers-1):                     # num_layer - 1 not included in count
                
           
                weights_of_curr_layer  = self.params['W'+ str(num_of_layer + 1)]
                biases_of_curr_layer   = self.params['b'+ str(num_of_layer + 1)]
                
                
               
                if ((self.num_layers - 1) <= 1):

                    scor_nextHidden = scor_Hidlayer
                    #print('Batchnorm false and Dropout false======================000==========================')
                else:
                    if self.use_batchnorm and (not self.use_dropout):
                        #print('Batchnorm true and Dropout false')
                        # With batch normalization
                        gamma = self.params['gamma'+ str(num_of_layer + 1)]
                        beta = self.params['beta'+ str(num_of_layer + 1)]

                        scor_nextHidden, l_params = self.affine_norm_relu_forward(scor_Hidlayer
			     							      ,weights_of_curr_layer
										      ,biases_of_curr_layer
										      ,gamma
										      ,beta
										      ,self.bn_params[num_of_layer]
							    		          )  

                        scor_Hidlayer = scor_nextHidden
                        feed_params[num_of_layer] = l_params
                    
                    if self.use_batchnorm and self.use_dropout: 
                       
                         # With batch normalization
                        gamma = self.params['gamma'+ str(num_of_layer + 1)]
                        beta = self.params['beta'+ str(num_of_layer + 1)]

                        scor_nextHidden, l_params = self.affine_norm_relu_drop_forward(scor_Hidlayer
			     							      ,weights_of_curr_layer
										      ,biases_of_curr_layer
										      ,gamma
										      ,beta
										      ,self.bn_params[num_of_layer]
										      ,self.dropout_param
							    		          )  

                        scor_Hidlayer = scor_nextHidden
                        feed_params[num_of_layer] = l_params

                    if (not self.use_batchnorm) and  self.use_dropout: 
                         print('Grad - Batchnorm false and Dropout true')
                         scor_nextHidden, l_params = self.affine_relu_drop_forward(scor_Hidlayer
								                  ,weights_of_curr_layer
								                  ,biases_of_curr_layer   
										  ,self.dropout_param
										)
                         scor_Hidlayer = scor_nextHidden
                         feed_params[num_of_layer] = l_params
                        
                    if (not self.use_batchnorm) and  (not self.use_dropout):
                         #print('Batchnorm false and Dropout false================================================')
                         #print('Not here')
                         # Without Batch Normalization
                         
                         scor_nextHidden, l_params = affine_relu_forward(scor_Hidlayer
     							      ,weights_of_curr_layer
							      ,biases_of_curr_layer
							     ) 
                    
                         scor_Hidlayer = scor_nextHidden
                         feed_params[num_of_layer] = l_params
                         
                   
                  
        last_hid_layer = self.num_layers - 1 
        last_params = None
        scores = None
       
        '''
        if self.use_batchnorm:
             scores, last_params = self.affine_norm_forward(scor_Hidlayer
                                              ,self.params['W'+ str(last_hid_layer + 1)]
                                              ,self.params['b'+ str(last_hid_layer + 1)]
                                              ,self.params['gamma'+ str(last_hid_layer + 1)]
                                              ,self.params['beta'+ str(last_hid_layer + 1)]
					      ,self.bn_params[last_hid_layer]                            #{'mode':'train'}

                                            )
        '''
       
       
        scores, last_params = affine_forward(scor_Hidlayer
                                              ,self.params['W'+ str(last_hid_layer + 1)]
                                              ,self.params['b'+ str(last_hid_layer + 1)]
                                            )
        #print('Shape of:', scores.shape)
        feed_params[last_hid_layer] = last_params
       

        #Computing the loss
        num_train = X.shape[0]
        exp_scores = np.exp(scores)
        prob = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        testProb = prob

               
        correct_scores_norm = prob[np.arange(prob.shape[0]),y]
        
        loss = np.sum(-np.log(correct_scores_norm))
        loss /=  num_train

        loss +=0.5*reg*(np.sum([np.sum(self.params['W'+ str(index)]**2) for index in range(1,self.num_layers)]))
        


        
     
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        #loss, grads = 0.0, {}
        grads = {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
       
        
        # Computing gradients
           
        for idx in range(1,self.num_layers + 1):
                grads['W'+ str(idx)] = np.zeros(self.params['W'+ str(idx)].shape)
                grads['b' + str(idx)] = np.zeros(self.params['b'+ str(idx)].shape)
                if self.use_batchnorm  and idx < self.num_layers:
                    grads['gamma' + str(idx)] = np.zeros(self.params['gamma'+ str(idx)].shape)
                    grads['beta' + str(idx)] = np.zeros(self.params['beta'+ str(idx)].shape)
                
        
        
        
        dL = prob                                         #(NxC)
        dL[np.arange(dL.shape[0]),y] -= 1     
        dL /= num_train 
        
        layers_derivatives = {} 
             
        '''
        if self.use_batchnorm:
              dL_dHidden, grads['W' + str(last_hid_layer + 1)] \
	      , grads['b' + str(last_hid_layer + 1)] \
	      , grads['gamma' + str(last_hid_layer)] \
	      , grads['beta' + str(last_hid_layer)] = self.affine_norm_backward(dL, feed_params[last_hid_layer])
        '''
        dL_dHidden, grads['W' + str(last_hid_layer + 1)],  grads['b' + str(last_hid_layer + 1)] =  affine_backward(dL, feed_params[last_hid_layer])
        
        
        for layer_num in range(last_hid_layer - 1 ,-1,-1):
            
                if self.use_batchnorm and (not self.use_dropout):
     
                     dL_dHidden ,grads['W' + str(layer_num + 1)] ,grads['b' + str(layer_num + 1)] ,grads['gamma' + str(layer_num + 1)] ,grads['beta' + str(layer_num + 1)] \
				= self.affine_norm_relu_backward(dL_dHidden, feed_params[layer_num])
                       
                     grads['W' + str(layer_num + 1) ] += reg * self.params['W' + str(layer_num + 1) ]  

                if self.use_batchnorm and  self.use_dropout:
                     print('Backward with norm and dropout True')
     
                     dL_dHidden \
                     ,grads['W' + str(layer_num + 1)] \
                     ,grads['b' + str(layer_num + 1)]            \
                     ,grads['gamma' + str(layer_num + 1)]        \
                     ,grads['beta' + str(layer_num + 1)] = self.affine_norm_relu_drop_backward(dL_dHidden, feed_params[layer_num])
                       
                     grads['W' + str(layer_num + 1) ] += reg * self.params['W' + str(layer_num + 1) ]  

                if (not self.use_batchnorm) and self.use_dropout:
                     print('Backward with dropout = True')
     
                     dL_dHidden \
                     ,grads['W' + str(layer_num + 1)],grads['b' + str(layer_num + 1)] = self.affine_relu_drop_backward(dL_dHidden, feed_params[layer_num])
                     
                    
                       
                     grads['W' + str(layer_num + 1) ] += reg * self.params['W' + str(layer_num + 1) ]  
     
                if (not self.use_batchnorm) and (not self.use_dropout):

                      print('Backward with dropout = 0')
  
                      dL_dHidden, grads['W' + str(layer_num + 1)], grads['b' + str(layer_num + 1)] = affine_relu_backward(dL_dHidden ,feed_params[layer_num])

                      grads['W' + str(layer_num + 1) ] += reg * self.params['W' + str(layer_num + 1) ] 
                     
               
                
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        
        
        return loss, grads
  
    def affine_norm_relu_forward(self, x, w, b, gamma, beta, bn_param = {}):
        """
        Perform inner_prod->batch_normalization->ReLu

	    Inputs:
	    - x: Input to the affine layer
	    - w, b: Weights for the affine layer

	    Returns a tuple of:
	    - out: Output from the ReLU
	    - cache: Object to give to the backward pass
        """
        a, fc_cache = affine_forward(x, w, b)
        norm, norm_cache =  batchnorm_forward(a, gamma, beta, bn_param)
        out, relu_cache = relu_forward(norm)
        cache = (fc_cache, norm_cache, relu_cache)
        return out, cache

    def affine_norm_relu_backward(self, dout, cache):
        
      
        fc_cache, norm_cache, relu_cache = cache
        da = relu_backward(dout, relu_cache)
        dnorm, dgamma, dbeta = batchnorm_backward(da, norm_cache)
        dx, dw, db = affine_backward(dnorm, fc_cache)
        return dx, dw, db, dgamma, dbeta


    def affine_norm_relu_drop_forward(self, x, w, b, gamma, beta, bn_param = {}, drop_param = {}):
	
        
        a, fc_cache = affine_forward(x, w, b)
        norm, norm_cache =  batchnorm_forward(a, gamma, beta, bn_param)
        relu, relu_cache = relu_forward(norm)
        out, drop_cache  = dropout_forward(relu, drop_param)
        cache = (fc_cache, norm_cache, relu_cache, drop_cache)
        return out, cache

 
    def affine_norm_relu_drop_backward(self, dout, cache):  
       
        fc_cache, norm_cache, relu_cache, drop_cache = cache
        ddrop = dropout_backward(dout , drop_cache)
        da = relu_backward(ddrop, relu_cache)
        dnorm, dgamma, dbeta = batchnorm_backward(da, norm_cache)
        dx, dw, db = affine_backward(dnorm, fc_cache)
        return dx, dw, db, dgamma, dbeta


    def affine_relu_drop_forward(self, x, w, b, drop_param = {}):
         
        a, fc_cache = affine_forward( x, w, b)
        #norm, norm_cache =  batchnorm_forward(a, gamma, beta, bn_param)
        relu, relu_cache = relu_forward(a)
        out, drop_cache = dropout_forward(relu , drop_param) 
        cache = (fc_cache, relu_cache, drop_cache)
        return out, cache

    def affine_relu_drop_backward(self, dout, cache):
         
        

        fc_cache, relu_cache, drop_cache = cache

        ddrop = dropout_backward(dout, drop_cache)

        drelu = relu_backward(ddrop, relu_cache)
        dx, dw, db = affine_backward(drelu, fc_cache)

        return dx, dw, db



    def forward_helper(params,batchnorm=False, dropout = False):
        pass
    def backward_helper(params, batchnorm=False, dropout = False):
        pass
        
    
       

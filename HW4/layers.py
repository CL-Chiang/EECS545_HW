from builtins import range
import numpy as np
import math


def fc_forward(x, w, b):
    """
    Computes the forward pass for a fully-connected layer.
    The input x has shape (N, d_in) and contains a minibatch of N
    examples, where each example x[i] has d_in element.
    Inputs:
    - x: A numpy array containing input data, of shape (N, d_in)
    - w: A numpy array of weights, of shape (d_in, d_out)
    - b: A numpy array of biases, of shape (d_out,)
    Returns a tuple of:
    - out: output, of shape (N, d_out)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Implement the forward pass. Store the result in the variable out. #
    ###########################################################################
    out = x @ w + b
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def fc_backward(dout, cache):
    """
    Computes the backward pass for a fully_connected layer.
    Inputs:
    - dout: Upstream derivative, of shape (N, d_out)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_in)
      - w: Weights, of shape (d_in, d_out)
      - b: Biases, of shape (d_out,)
    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d_in)
    - dw: Gradient with respect to w, of shape (d_in, d_out)
    - db: Gradient with respect to b, of shape (d_out,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the affine backward pass.                               #
    ###########################################################################
    dx = dout @ w.T
    dw = x.T @ dout
    db = np.sum(dout, axis= 0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).
    Input:
    - x: Inputs, of any shape
    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Implement the ReLU forward pass.                                  #
    ###########################################################################
    out = x * (x>0)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).
    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout
    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Implement the ReLU backward pass.                                 #
    ###########################################################################
    dx = np.array(dout, copy = True)
    dx[cache <=0] =0
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def conv_forward(x, w):
    """
    The input consists of N data points, each with C channels, height H and
    width W. We filter each input with F different filters, where each filter
    spans all C channels and has height H' and width W'.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, H', W')
    Returns a tuple of:
    - out: Output data, of shape (N, F, HH, WW) where H' and W' are given by
      HH = H - H' + 1
      WW = W - W' + 1
    - cache: (x, w)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass based on the definition  #
    # of Y in Q1(c).                                                          #
    ###########################################################################
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_p = H - HH + 1
    W_p = W - WW + 1
    out = np.zeros((N, F, H_p, W_p))

    # F, C, H', W' => newaxis, C, H', W', F
    w_T = w[:,:,:,:,np.newaxis].transpose(4, 1, 2, 3, 0)

    for h_ in range(H_p):
      for w_ in range(W_p):
          out[:,:,h_,w_] = np.sum(x[:,:,h_:h_+HH,w_:w_+WW, np.newaxis] * w_T, axis = (1,2,3))
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w)
    return out, cache


def conv_backward(dout, cache):
    """
    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w) as in conv_forward
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    """
    dx, dw = None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w = cache
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_p = H - HH + 1
    W_p = W - WW + 1
    dw = np.zeros_like(w) # w: F, C, HH, WW
    dx = np.zeros_like(x)
    # out: N, F, H_p, W_p   x: N, C, H, W

    # N, F, H_p, W_p => F, newaxis, N, H_P, W_p
    dout_T = dout[:,:,:,:,np.newaxis].transpose(1,4,0,2,3)
    # N, C, H, W => newaxis, C, N, H, W
    x_T = x[:,:,:,:,np.newaxis].transpose(4,1,0,2,3)
    for i in range(HH):
      for j in range(WW):
        dw[:,:,i,j] = np.sum(x_T[:,:,:,i:i+H_p, j:j+W_p] * dout_T, axis = (2,3,4))
    
    #out N, F, H', W'   W: F, C, HH, WW
    doutp = np.pad(dout, ((0,), (0,), (WW-1,), (HH-1, )), 'constant')
    w_ = np.zeros_like(w)
    for i in range(HH):
        for j in range(WW):
            w_[:,:,i,j] = w[:,:,HH-i-1,WW-j-1]
    
    # dout_T  N, new, H', W', F   w_T: new, C, HH, WW, F
    w_T = w_[:,:,:,:, np.newaxis].transpose(4,1,2,3,0)
    dout_T = doutp[:,:,:,:,np.newaxis].transpose(0,4,2,3,1)
    for i in range(H):
      for j in range(W):
        dx[:,:,i,j] = np.sum(dout_T[:, :, i:i+HH, j:j+WW, :] * w_T, axis = (2,3,4))

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw


def max_pool_forward(x, pool_param):
    """
    A naive implementation of the forward pass for a max-pooling layer.
    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions
    No padding is necessary here and we can assume that the dimension of
    input and stride will not cause problem here. Output size is given by
    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    N,C,H,W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    H_p = int(1 + (H - pool_height) / stride)
    W_p = int(1 + (W - pool_width) / stride)
    out = np.zeros((N, C, H_p, W_p))
    Ind = np.zeros((N, C, H_p, W_p, pool_height * pool_width))
    for h in range(H_p):
      for w in range(W_p):
        hstart = h * stride
        hend = hstart + pool_height
        wstart = w * stride
        wend = wstart + pool_width
        out[:, :, h, w] = np.max(x[:,:, hstart:hend, wstart:wend], axis = (2,3))
        for n in range(N):
          for c in range(C):
              ht, wt = np.where(x[n, c, hstart:hend, wstart:wend] == out[n,c,h,w])
              Ind[n, c, h, w, ht * pool_width + wt] = 1
              #Ind[n, c, hstart,wstart] = ht * pool_height+ wt
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, Ind, pool_param)
    return out, cache


def max_pool_backward(dout, cache):
    """
    A naive implementation of the backward pass for a max-pooling layer.
    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.
    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    x, Ind, pool_param = cache
    dx = np.zeros_like(x)
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    stride = pool_param['stride']
    N,C,H_p,W_p = dout.shape
    
    for h in range(H_p):
      for w in range(W_p):
        hstart = h * stride
        hend = hstart + pool_height
        wstart = w * stride
        wend = wstart + pool_width
        tempdout = dout[:,:,h,w]
        tempdx = (np.repeat(tempdout[:,:,np.newaxis], pool_height * pool_width, axis = 2) * Ind[:,:,h, w,:])
        dx[:, :, hstart:hend, wstart:wend] += tempdx.reshape(N,C,pool_height, pool_width)
    #dx = dx * x
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.
    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the j-th
      class for the i-th input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
    Returns a tuple of:
    - loss: Scalar giving the cross-entropy loss averaged over N samples.
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Implement the softmax loss
    ###########################################################################
    N, C = x.shape
    loss = np.zeros(N)
    p = np.exp(x)
    psum = np.sum(p, axis = 1, keepdims = True)
    p = p / psum
    dx = p
    for i in range(N):
      loss[i] = -np.log(p[i,y[i]])
      dx[i,y[i]] -= 1
    loss = np.sum(loss) / N
    dx /= N

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx

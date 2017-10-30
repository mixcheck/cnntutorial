from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def activation_summary(x):
    """Helper function to create summaries for activations.
    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.
    Args:
        x: Tensor
    Returns:
        nothing
    """
    tf.summary.histogram(x.op.name + '/activations', x)
    tf.summary.scalar(x.op.name + '/sparsity', tf.nn.zero_fraction(x))    
    
def get_initializer(init_type, **params):
    """Helper function to get initialization function.
    Args:
        init_type: type of initialization (constant, normal, uniform)
        params: parameters for initialization function
    Returns:
        initialization function    
    """
    if init_type == 'constant':
        init_func = tf.constant_initializer(params['value'])
    elif init_type == 'normal':
        init_func = tf.truncated_normal_initializer(stddev=params['stddev'], dtype=tf.float32)
    elif init_type == 'uniform':
        init_func = tf.random.uniform_initializer(minval=-params['value'], maxval=params['value'])
    else:
        raise ValueError('Unknown initialization function type')
        
    return init_func    
    
def create_variable(name, shape, initializer, weight_decay=None):
    """Helper function to create a Variable.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
        weight_decay: weight decay factor, if it is not None apply weight decay
    Returns:
        Variable Tensor
    """
    var = tf.get_variable(name, shape, initializer=initializer, dtype=tf.float32)
    
    if weight_decay is not None:
        weight_decay = tf.multiply(tf.nn.l2_loss(var), weight_decay, name='weight_loss')
        tf.add_to_collection('losses', weight_decay)
    
    return var

def get_conv2D_layer(inp, inp_dim, out_dim, filter_size, stride, initializer, weight_decay, 
                    activation_func='relu', scope=None, write_summary=True):
    """Helper function to create 2D convolutional layer
    Args:
        inp: input layer
        inp_dim: dimmension of input layer
        out_dim: dimmension of this layer
        filter_size: filter size
        stride: stride
        initializer: weight initializer function
        weight_decay: weight decay factor applying l2 loss for weight
        activation_func: activation function (relu, tanh)
        scope: variable scope 
        write_summary: whether to write summary of weights in the tensorboard
    Returns:
        convolutional layer
    """
    # Create weight (filter) and bias for convolution
    filters = create_variable('weights', [filter_size, filter_size, inp_dim, out_dim],
                            initializer, weight_decay)
    biases = create_variable('biases', [out_dim], initializer)
    
    # apply convolution and add bias
    conv = tf.nn.conv2d(inp, filters, [1, 1, 1, 1], padding='VALID', name=scope.name+'_out')
    pre_activation = tf.nn.bias_add(conv, biases, name=scope.name+ '_bias_out')
    
    # apply activation function
    if activation_func == 'relu':
        conv_out = tf.nn.relu(pre_activation, name=scope.name+'_relu')
    elif activation_func == 'tanh':
        conv_out = tf.tanh(pre_activation, name=scope.name+'_tanh')
    else:
        conv_out = pre_activation
    
    # write 
    if write_summary:
        activation_summary(conv_out)
    
    return conv_out

def get_fully_connected_layer(inp, inp_dim, out_dim, initializer, weight_decay, activation_func='relu', 
                              flatten=False, batch_size=None, scope=None, write_summary=True):
    """Helper function to create fully connected layer
    Args:
        inp: input layer
        inp_dim: dimmension of input layer
        out_dim: dimmension of this layer
        initializer: weight initializer function
        weight_decay: weight decay factor applying l2 loss for weight
        activation_func: activation function (relu, tanh)
        scope: variable scope 
        write_summary: whether to write summary of weights in the tensorboard
    Returns:
        convolutional layer
    """
    # flatten the input layer
    if flatten:
        if batch_size is None:
            raise ValueError('batch_size is required when an inpu layer is flattened')
        else:
            reshape = tf.reshape(inp, [batch_size, -1])
            dim = reshape.get_shape()[1].value
    else:
        reshape = inp
        dim = inp_dim
        
    # Create weight and bias for convolution        
    weights = create_variable('weights', [dim, out_dim],
                            initializer, weight_decay)
    biases = create_variable('biases', [out_dim], initializer)
    
    # apply linear operation and add bias
    fc = tf.matmul(reshape, weights, name=scope.name+'_out')
    pre_activation = tf.nn.bias_add(fc, biases, name=scope.name+'_bias_out')
    
    # apply activation function
    if activation_func == 'relu':
        fc_out = tf.nn.relu(pre_activation, name=scope.name+'_relu')
    elif activation_func == 'tanh':
        fc_out = tf.tanh(pre_activation, name=scope.name+'_tanh')
    else:
        fc_out = pre_activation
    
    # write 
    if write_summary:
        activation_summary(fc_out)
    
    return fc_out
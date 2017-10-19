from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
from math import ceil
import sys

import numpy as np
import tensorflow as tf
slim = tf.contrib.slim

VGG_MEAN = [103.939, 116.779, 123.68]

class FCN:

    def __init__(self):
        self.wd = 5e-4

    def build(self, rgb, net_type='fcn_32s', train=False, num_classes=20, 
            random_init_fc8=False, debug=False):
        """
        Build the VGG model using loaded weights
        Parameters
        ----------
        rgb: image batch tensor
            Image in rgb shap. Scaled to Intervall [0, 255]
        net:type: Network type [fcn_32s, fcn_16s, fcn_8s, deconvNet]
        train: bool
            Whether to build train or inference graph
        num_classes: int
            How many classes should be predicted (by fc8)
        random_init_fc8 : bool
            Whether to initialize fc8 layer randomly.
            Finetuning is required in this case.
        debug: bool
            Whether to print additional Debug Information.
        """
        
        # Convert RGB to BGR
        
        with tf.name_scope('Processing'):

            red, green, blue = tf.split(rgb, 3, 3)
            # assert red.get_shape().as_list()[1:] == [224, 224, 1]
            # assert green.get_shape().as_list()[1:] == [224, 224, 1]
            # assert blue.get_shape().as_list()[1:] == [224, 224, 1]
            bgr = tf.concat([
                blue - VGG_MEAN[0],
                green - VGG_MEAN[1],
                red - VGG_MEAN[2],
            ], 3)

            if debug:
                bgr = tf.Print(bgr, [tf.shape(bgr)],
                               message='Shape of input image: ',
                               summarize=4, first_n=1)
        
        with tf.variable_scope('vgg_16', values=[bgr]) as sc:
            with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d]):

                self.conv1_2 = slim.repeat(bgr, 2, slim.conv2d, 64, [3,3], scope='conv1')
                self.pool1 = slim.max_pool2d(self.conv1_2, [2,2], scope='pool1')
                self.conv2_2 = slim.repeat(self.pool1, 2, slim.conv2d, 128, [3,3], scope='conv2')
                self.pool2 = slim.max_pool2d(self.conv2_2, [2,2], scope='pool2')
                self.conv3_3 = slim.repeat(self.pool2, 3, slim.conv2d, 256, [3,3], scope='conv3')
                self.pool3 = slim.max_pool2d(self.conv3_3, [2,2], scope='pool3')
                self.conv4_3 = slim.repeat(self.pool3, 3, slim.conv2d, 512, [3,3], scope='conv4')
                self.pool4 = slim.max_pool2d(self.conv4_3, [2,2], scope='pool4')
                self.conv5_3 = slim.repeat(self.pool4, 3, slim.conv2d, 512, [3,3], scope='conv5')
                self.pool5 = slim.max_pool2d(self.conv5_3, [2,2], scope='pool5')
                self.fc6 = slim.conv2d(self.pool5, 4096, [7,7], padding='SAME', scope='fc6')
                self.fc6 = slim.dropout(self.fc6, 0.5, is_training=train, scope='dropout6')
                self.fc7 = slim.conv2d(self.fc6, 4096, [1,1], padding='SAME', scope='fc7')
                self.fc7 = slim.dropout(self.fc7, 0.5, is_training=train, scope='dropout7')
                self.score_fr = slim.conv2d(self.fc7, num_classes, [1,1], padding='SAME',
                        activation_fn=None, normalizer_fn=None, scope='score_fr')

        self.pred = tf.argmax(self.score_fr, axis=3)
        if net_type == 'fcn_32s':
            self.upscore = self._upscore_layer(self.score_fr, 
                    output_shape=tf.shape(bgr),
                    out_dims=num_classes,
                    debug=debug,
                    name='up', factor=32)
        elif net_type == 'fcn_16s':
            # TODO: implement fcn_16s
            self.upscore2 = self._upscore_layer(self.score_fr,
                    output_shape = tf.shape(self.pool4),
                    out_dims = num_classes,
                    debug = debug, name = 'upscore2',
                    factor = 2)
            self.score_pool4 = slim.conv2d(self.pool4, num_classes, [1,1], padding='SAME',
                    activation_fn=None, normalizer_fn=None, scope='score_pool4')
            self.fuse_pool4 = tf.add(self.upscore2, self.score_pool4)
            self.upscore = self._upscore_layer(self.fuse_pool4,
                    output_shape = tf.shape(bgr),
                    out_dims = num_classes,
                    debug = debug, name = 'upscore32',
                    factor = 16)
        elif net_type == 'fcn_8s':
            # TODO: implement fcn_8s
            TODO = True
        else:
            TODO = True

        self.pred_up = tf.argmax(self.upscore, axis=3)

    def _get_bilinear_filter(self, size):
        """
        Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
        """
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)

    def _get_bilinear_weights(self, factor, num_classes):
        """
        Create weights matrix for transposed convolution with bilinear filter
        initialization.
        """
        filter_size = 2 * factor - factor % 2

        weights = np.zeros((filter_size,
                            filter_size,
                            num_classes,
                            num_classes), dtype=np.float32)

        bilinear = self._get_bilinear_filter(filter_size)
        for i in range(num_classes):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)

    def _upscore_layer(self, bottom, output_shape,
                       out_dims, name, debug, factor=2):

        # inp_shape = (batch, height, width, in_dims)
        with tf.variable_scope(name):

            # Compute parameters for deconvolution 
            # Obtain bilinear filter weight (filter_size, filter_size, out_dims, out_dims)
            weights = self._get_bilinear_weights(factor, out_dims)

            # Obtain output shape
            out_shape = tf.stack([output_shape[0], output_shape[1], 
                output_shape[2], out_dims])
            # Obtain stride parameter
            strides = [1, factor, factor, 1]
            deconv = tf.nn.conv2d_transpose(bottom, weights, out_shape,
                                strides, padding='SAME')

        return deconv
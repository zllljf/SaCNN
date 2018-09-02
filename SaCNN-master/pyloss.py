#!/usr/bin/env python
# -*- coding: utf-8 -*-

caffe_root = '/home/zll/OPT/caffe/'
import sys
sys.path.insert(0, caffe_root + 'python')
import caffe
import numpy as np

#add python layer in caffe
class MseMetricLayer(caffe.Layer):
    """
    Compute the MSE in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sqrt(np.sum(self.diff**2) / bottom[0].num)

    def backward(self, top, propagate_down, bottom):
        # metric do not need backward
        pass


class MaeMetricLayer(caffe.Layer):
    """
    Compute the MAE in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute distance.")

    def reshape(self, bottom, top):
        # check input dimensions match
        if bottom[0].count != bottom[1].count:
            raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.diff = np.zeros_like(bottom[0].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(1)

    def forward(self, bottom, top):
        self.diff[...] = bottom[0].data - bottom[1].data
        top[0].data[...] = np.sum(np.abs(self.diff)) / bottom[0].num

    def backward(self, top, propagate_down, bottom):
        # metric do not need backward
        pass
from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow
import numpy as np
import scipy.io
import re

"""
Written to convert the Inception v1 checkpoint file to matlab data files. 

Note that Inception Layer 5b, Branch 2, Convolution 0a and 0b are both incorrectly labeled 0a. This causes 0b to overwrite 0a.
To get around this issue, consider removing the '\dx\d' regex to obtain both sets of weights
"""

# tf.reset_default_graph()

# saver = tf.train.import_meta_graph('TFtransformer_files/imagenet_inception_v1.ckpt.meta')

# with tf.Session() as sess:
	# new_saver = tf.train.import_meta_graph('TFtransformer_files/imagenet_inception_v1.ckpt.meta')
	# new_saver.restore(sess, tf.train.latest_checkpoint('C:/Users/Eugene/Documents/UCLA/Research/TFtransformer_files/imagenet_inception_v1.ckpt'))
	# new_saver.restore(sess,'C:/Users/Eugene/Documents/UCLA/Research/TFtransformer_files/imagenet_inception_v1.ckpt')
	# print("Model restored.")
	
# chkp.print_tensors_in_checkpoint_file('C:/Users/Eugene/Documents/UCLA/Research/TFtransformer_files/imagenet_inception_v1.ckpt', tensor_name='', all_tensors=False, all_tensor_names=True)

# dir = 'C:/Users/Eugene/Documents/UCLA/Research/TFtransformer_files/imagenet_inception_v1.ckpt'
dir = 'C:/Users/Eugene/Documents/UCLA/Research/inception_v1/inception_v1.ckpt'

reader = pywrap_tensorflow.NewCheckpointReader(dir)
var_to_shape_map = reader.get_variable_to_shape_map()
for key in sorted(var_to_shape_map):
	name = re.sub('InceptionV1/', '', key)
	name = re.sub('/', '_', name)
	name = re.sub('Conv2d', 'conv', name)
	name = re.sub('\dx\d_', '', name)
	name = re.sub('BatchNorm_moving_mean', 'batchmean', name)
	name = re.sub('BatchNorm_moving_variance', 'batchvar', name)
	name = re.sub('BatchNorm_beta', 'batchbeta', name)
	scipy.io.savemat(name, dict(weights=reader.get_tensor(key)))

from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from urllib import request
from tensorflow.python.tools import inspect_checkpoint as chkp
from tensorflow.python import pywrap_tensorflow
from tensorflow.contrib import slim
import imagenet
import inception_v1
import inception_preprocessing
import numpy as np
import scipy.io
import re

# tf.reset_default_graph()
# sess = tf.Session()
# saver = tf.train.import_meta_graph('TFtransformer_files/imagenet_inception_v1.ckpt.meta')
# saver.restore(sess,'C:/Users/Eugene/Documents/UCLA/Research/TFtransformer_files/imagenet_inception_v1.ckpt')
# print("Model restored.")
# graph = tf.get_default_graph()


image_size = inception_v1.inception_v1.default_image_size
checkpoints_dir = 'C:/Users/Eugene/Documents/UCLA/Research/TFtransformer_files'

with tf.Graph().as_default():
    url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
    image_string = request.urlopen(url).read()
    image = tf.image.decode_jpeg(image_string, channels=3)
    processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
    processed_images  = tf.expand_dims(processed_image, 0)
    
    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
        logits, _ = inception_v1.inception_v1(processed_images, num_classes=1001, is_training=False)
    probabilities = tf.nn.softmax(logits)
    
    init_fn = slim.assign_from_checkpoint_fn(
        os.path.join(checkpoints_dir, 'inception_v1.ckpt'),
        slim.get_model_variables('InceptionV1'))
    
    with tf.Session() as sess:
        init_fn(sess)
        np_image, probabilities = sess.run([image, probabilities])
        probabilities = probabilities[0, 0:]
        sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
        
    plt.figure()
    plt.imshow(np_image.astype(np.uint8))
    plt.axis('off')
    plt.show()

    names = imagenet.create_readable_names_for_imagenet_labels()
    for i in range(5):
        index = sorted_inds[i]
        print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))



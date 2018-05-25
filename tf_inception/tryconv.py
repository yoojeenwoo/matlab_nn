import numpy as np
import tensorflow as tf
from scipy.io import loadmat

x = loadmat('image/Image01.mat')
x_image = x['image']

x_image = np.asarray(x_image)

f_weight = loadmat('../inception_v1_float_weights/conv_1a_weights.mat')
f_weight = np.asarray(f_weight['weights'])

x_in = tf.placeholder(tf.float32, shape=(1, 224, 224, 3))
conv_weight = tf.placeholder(tf.float32, shape=(7, 7, 3, 64))
y = tf.nn.conv2d(x_in, conv_weight, strides=(1, 1, 1, 1), padding="SAME")

sess = tf.Session()
temp = sess.run(y, feed_dict={x_in: x_image, conv_weight: f_weight})
print('---conv output---')
print(temp[0][0][0][0])

print('---image input shape---')
temp_x = np.zeros((1,7,7,3), dtype='float')
temp_x[0][3:7,3:7,:] = x_image[0][:4,:4,:]
# temp_x = x_image[0][:7,:7,:]
print(x_image[0][:7,:7,:].shape)

print('---weight matrix shape---')
temp_w = f_weight[:,:,:,0]
print(f_weight[:,:,:,0].shape)

print('element wise multiplication result:')
print(np.sum(np.multiply(temp_x, temp_w)))



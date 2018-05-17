from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from urllib import request
from tensorflow.contrib import slim
import imagenet
import inception_v1
import inception_preprocessing
import numpy as np

# tf.reset_default_graph()
# sess = tf.Session()
# saver = tf.train.import_meta_graph('TFtransformer_files/imagenet_inception_v1.ckpt.meta')
# saver.restore(sess,'C:/Users/Eugene/Documents/UCLA/Research/TFtransformer_files/imagenet_inception_v1.ckpt')
# print("Model restored.")
# graph = tf.get_default_graph()


image_size = inception_v1.inception_v1.default_image_size
checkpoints_dir = 'C:/Users/Eugene/Documents/UCLA/Research/TFtransformer_files'

with tf.Graph().as_default():
	# url = 'https://upload.wikimedia.org/wikipedia/commons/7/70/EnglishCockerSpaniel_simon.jpg'
	# image_string = request.urlopen(url).read()
	
	image_dir = "C:/Users/Eugene/Documents/UCLA/Research/imagenet-data/"
	filenames = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
	filename_queue = tf.train.string_input_producer(filenames)
	# filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once("C:/Users/Eugene/Documents/UCLA/Research/imagenet-data/*.jpg"))
	image_reader = tf.WholeFileReader()
	_, image_file = image_reader.read(filename_queue)
	
	print("Image File Read")
	
	image = tf.image.decode_jpeg(image_file, channels=3)
	processed_image = inception_preprocessing.preprocess_image(image, image_size, image_size, is_training=False)
	processed_images  = tf.expand_dims(processed_image, 0)
	
	print("Images Processed")
	
	# Create the model, use the default arg scope to configure the batch norm parameters.
	with slim.arg_scope(inception_v1.inception_v1_arg_scope()):
		logits, _ = inception_v1.inception_v1(processed_images, num_classes=1001, is_training=False)
	probabilities = tf.nn.softmax(logits)
	
	print ("Neural Network Built")
	
	init_fn = slim.assign_from_checkpoint_fn(
		os.path.join(checkpoints_dir, 'inception_v1.ckpt'),
		slim.get_model_variables('InceptionV1'))
	
	print ("Checkpoint Loaded")
	with tf.Session() as sess:
		# Required to get the filename matching to run.
		tf.global_variables_initializer().run()
		tf.local_variables_initializer().run()
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
	
		init_fn(sess)
		# np_image, probabilities = sess.run([image, probabilities])
		np_image, probabilities = sess.run([image, probabilities])
		probabilities = probabilities[0, 0:]
		sorted_inds = [i[0] for i in sorted(enumerate(-probabilities), key=lambda x:x[1])]
		
		coord.request_stop()
		coord.join(threads)
	
	plt.figure()
	plt.imshow(np_image.astype(np.uint8))
	plt.axis('off')
	plt.show()

	names = imagenet.create_readable_names_for_imagenet_labels()
	for i in range(5):
		index = sorted_inds[i]
		print('Probability %0.2f%% => [%s]' % (probabilities[index] * 100, names[index]))
	


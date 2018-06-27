# import pandas as pd
# import os
import csv
import numpy as np
import tensorflow as tf

# DIR_PATH = os.path.dirname(os.path.abspath(__file__)) 
DIR_PATH = '.'
TRAIN_DATA = DIR_PATH + '/data/train.csv'
TEST_DATA = DIR_PATH + '/data/test.csv'
SAVER_FILE = DIR_PATH + '/save/mnist.ckpt'

dataset = {}
weights = {}
biases = {}

def get_MaxIndex(array_val):
	maxIndex = 0
	for i in range(len(array_val)):
		if array_val[i] > array_val[maxIndex]:
			maxIndex = i

	return maxIndex

def Gen_Dataset():
	img_data = np.genfromtxt(TRAIN_DATA, delimiter=',')
	# print(img_data.shape)

	# delete row 0
	img_data = np.delete(img_data, 0, 0)
	# print(img_data.shape)

	# get colume 0 as array
	img_label = img_data[:, 0].astype(int)
	# print(img_label[0:4])

	# convert label to 2-dim array, label mapping 0 ~ 9
	img_label_2d = np.zeros((img_label.size, 10))
	# print(img_label_2d.shape)

	for idx, row in enumerate(img_label_2d):
		row[img_label[idx]] = 1
	# print(img_label_2d[0:4])


	# delete colume 0
	img_data = np.delete(img_data, np.s_[:1], 1)
	# print(img_data.shape)

	# normailize
	img_data /= 255.0

	dataset['train_img'] = img_data
	dataset['train_label'] = img_label_2d


	# Test Data
	img_data = np.genfromtxt(TEST_DATA, delimiter=',')
	# delete row 0
	img_data = np.delete(img_data, 0, 0)
	# normailize
	img_data /= 255.0
	dataset['test_img'] = img_data


def add_hiddenLayers(input_x, weights, biases):
	hidden_layer = tf.add( tf.matmul(input_x, weights['w1']), biases['b1'])
	hidden_layer = tf.nn.relu(hidden_layer) # activation function
	output_layer = tf.add( tf.matmul(hidden_layer, weights['w2']), biases['b2'])
	return output_layer

def init_nn():
	global weights_optimizer, loss, accuracy, x, y, predictions
	input_size = 784 #28x28
	hidden_size = 196 #50 #hidden layer
	output_size = 10 #class size
	weights['w1'] = tf.Variable(tf.random_normal([input_size, hidden_size]))
	weights['w2'] = tf.Variable(tf.random_normal([hidden_size, output_size]))
	biases['b1'] = tf.Variable(tf.random_normal([hidden_size]))
	biases['b2'] = tf.Variable(tf.random_normal([output_size]))

	# set graph to read outside data
	x = tf.placeholder(tf.float32, [None, input_size])
	y = tf.placeholder(tf.float32, [None, output_size])

	# add hidden layers
	predictions = add_hiddenLayers(x, weights, biases)

	# loss function
	loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=y) )

	# learning rate
	weights_optimizer = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

	# accuracy eval
	current_prediction = tf.equal( tf.argmax(predictions, 1), tf.argmax(y, 1) )
	accuracy = tf.reduce_mean( tf.cast(current_prediction, tf.float32) )


def run_nn(batch_size, run_epoch, isRestore):
	# global weights_optimizer
	train_size = dataset['train_img'].shape[0]
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run( tf.global_variables_initializer() )

		if not isRestore:
			total_batches = int(train_size / batch_size)
			for epoch in range(run_epoch):
				avg_loss = 0.0
				for i in range(total_batches):
					batch_mask = np.random.choice(train_size, batch_size)
					x_batch = dataset['train_img'][batch_mask]
					y_batch = dataset['train_label'][batch_mask]
					_, c = sess.run( [weights_optimizer, loss], feed_dict={x: x_batch, y: y_batch} )
					avg_loss += c / total_batches

				if epoch % 10 == 0:
					print('Epoch:', '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss) )
			print('train finished')

			print('===save model===')
			saver.save(sess, SAVER_FILE)
		else:
			saver.restore(sess, SAVER_FILE)

		# print('===eval accuracy===')
		# batch_mask = np.array([0])
		# count = 0
		# total_size = dataset['train_img'].shape[0]
		# for i in range(total_size):
		# 	x_batch = dataset['train_img'][batch_mask]
		# 	y_batch = dataset['train_label'][batch_mask]
		# 	batch_mask += 1
		# 	count = count + accuracy.eval({x: x_batch, y: y_batch})
		# count = count / total_size
		# print('trained accuracy=', count)

		print('===predict test image===')
		batch_mask = np.array([0])
		csv_file = open('test_result.csv', 'w', newline='')
		result_lists = []
		result_lists.append(['ImageId', 'Label'])
		for i in range(dataset['test_img'].shape[0]):
			x_batch = dataset['test_img'][batch_mask]
			# print(x_batch.shape)
			result = predictions.eval(feed_dict={x: x_batch})
			# print( result )
			label = get_MaxIndex(result[0])
			# print('label=', label)
			data = []
			data.append(i+1)
			data.append(label)
			result_lists.append(data)
			batch_mask += 1
		w = csv.writer(csv_file)
		w.writerows(result_lists)
		csv_file.close()
		

if __name__ == "__main__":
	# print(DIR_PATH)
	# FILE_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../MNIST/'

	print("===loading data===")
	Gen_Dataset()
	np_array = dataset['train_label']
	print(np_array.shape)
	print(np_array[:4])

	print("===start training===")
	init_nn()
	run_nn(100, 500, True)




from __future__ import print_function
import numpy
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import tensorflow as tf
from sklearn.metrics import confusion_matrix

def read_dataset():
	df = np.genfromtxt("static_0.csv",delimiter=',')
	X = np.array(df[:,:17])
	y = np.array(df[:,17])
	Z = np.array(df[:,17])
		
	#Encode the dependent variables
	encoder = LabelEncoder()
	encoder.fit(y)
		
	y = encoder.transform(y)
	#	print(y)
	Y=np.zeros((y.shape[0],5))
	for i in range(0,Y.shape[0]):
		Y[i]=np.array([(1 if j==y[i] else 0) for j in range(0,5)])
	return (X,Y,Z)

x,y,z = read_dataset()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.01, random_state = 415)

# Parameters
learning_rate = 0.003
training_epochs = 100
batch_size = 50
display_step = 1
num_train_examples = train_x.shape[0]
num_test_examples = test_x.shape[0]


n_hidden_1 = 300 # 1st layer number of neurons
n_hidden_2 = 300 # 2nd layer number of neurons
#n_hidden_3 = 200 # 3rd layer number of neurons
n_input = x.shape[1]
n_classes = y.shape[1]


# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])


# Store layers weight & bias
weights = {
	'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
	'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#	'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
	'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
	'b1': tf.Variable(tf.random_normal([n_hidden_1])),
	'b2': tf.Variable(tf.random_normal([n_hidden_2])),
#	'b3': tf.Variable(tf.random_normal([n_hidden_3])),
	'out': tf.Variable(tf.random_normal([n_classes]))
}


# Create model
def multilayer_perceptron(x):
	# Hidden fully connected layer with 60 neurons
	layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
 #   layer_1 = tf.nn.relu(layer_1)
	# Hidden fully connected layer with 60 neurons
	layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
 #   layer_2 = tf.nn.relu(layer_2)
 #   layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
 #   layer_3 = tf.nn.relu(layer_3)
	# Output fully connected layer with a neuron for each class
	out_layer = tf.add(tf.matmul(layer_2 , weights['out']) , biases['out'])
	return out_layer
# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)


	# Training cycle
	for epoch in range(training_epochs):
		avg_cost = 0.
		#if epoch>200:
		#	learning_rate=0.0005
		total_batch = int(num_train_examples/batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_x, batch_y = train_x[batch_size*i:batch_size*(i+1)], train_y[batch_size*i:batch_size*(i+1)]
			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
															Y: batch_y})
			# Compute average loss
			avg_cost += c / total_batch
		# Display logs per epoch step
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost={:.2f}".format(avg_cost))
	print("Optimization Finished!")
	np.set_printoptions(threshold='nan')
	#np.savetxt("h10.001.csv", sess.run([weights['h1']]),fmt='%5s',delimiter=',')
	#np.savetxt("weights0.001.csv", sess.run([weights]),fmt='%5f',delimiter=',')
	save_path = saver.save(sess, "/home/het3/InterIIT/model3.ckpt")
	print("Model saved in file: %s" % save_path)
	# Test model3
	pred = tf.nn.softmax(logits)  # Apply softmax to logits
	#print(z)
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
	# Calculate accuracy
	#con = tf.confusion_matrix(labels=z, predictions=correct_prediction)
	#print(con.eval()) 
	#con = tf.Print(con, [con], message="This is a: ")
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Accuracy:", accuracy.eval({X: test_x, Y: test_y}))

#print(confusion_matrix(z, correct_prediction))
#print(confusion_matrix([2, 1, 4, 5, 3, 1, 1],
#	                   [1, 1, 4, 5, 3, 1, 2]))
from __future__ import print_function
import numpy
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
import tensorflow as tf
import serial
import cv2
import operator 

arduinoData = serial.Serial('/dev/ttyACM0', 115200)
n_hidden_1 = 300 # 1st layer number of neurons
n_hidden_2 = 300 # 2nd layer number of neurons
#n_hidden_3 = 200 # 3rd layer number of neurons
n_input = 17   #x.shape[1]
n_classes = 43        #y.shape[1]

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

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    saver.restore(sess, "/home/gopa/Documents/InterIIT/PS1_model/model1.ckpt")
    while True:
	count=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	for q in range(10):
		while (arduinoData.inWaiting()==0): #Wait here until there is data
		    pass #do nothing
		arduinoString = arduinoData.readline() #read the line of text from the serial port
		dataArray = arduinoString.split(',')   #Split it into an array called dataArray
		# the a matrix would come from pyserial
		if len(dataArray)==17:
		    print(dataArray)
		    a = np.matrix(dataArray)
		    a = np.float32(a)
		    #print (multilayer_perceptron(a).eval())
		    pred = tf.nn.softmax(multilayer_perceptron(a))
		    b=pred.eval()
		    for i in range(0,43):
			if b[0][i]==1.0 :
			    count[i]+=1
			    break
	signal=count.index(max(count))
	signal+=1	
	img = cv2.imread("Symbol_images/"+str(signal)+".png",1)
	cv2.imshow("Symbol",img)
	cv2.waitkey(1)

#CNN MODEL FOR SS PREDICTION
#Code adapted from original by Jes Frellsen

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#1.load the data:

data = np.load('cullpdb+profile_6133_ss3-preprocessed.npz') #compressed binary file of the data in NumPy format
X_train = data['X_train']
y_train = data['y_train']
X_validation = data['X_validation']
y_validation = data['y_validation']
X_test = data['X_test']
y_test = data['y_test']

#2. Define the model/graph.

# Input and output:
X = tf.placeholder(tf.float32, [None, 700, 44], name="X")
y = tf.placeholder(tf.float32, [None, 700, 4], name='y')

# Model:
filter_width = 11
filter_input_size = 44
filter_channels = 40
final_filter_channels = 4
W1 = tf.get_variable(name="W1", shape=[filter_width, filter_input_size, filter_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())
b1 = tf.get_variable("b1",[filter_channels], initializer=tf.random_normal_initializer())
W2 = tf.get_variable(name="W2", shape=[filter_width, filter_input_size, filter_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())
b2 = tf.get_variable("b2",[filter_channels], initializer=tf.random_normal_initializer())
W3 = tf.get_variable(name="W3", shape=[filter_width, filter_input_size, final_filter_channels], initializer=tf.contrib.layers.xavier_initializer_conv2d())
b3 = tf.get_variable("b3",[final_filter_channels], initializer=tf.random_normal_initializer())

# First convolutional layer
conv1 = tf.nn.conv1d(value=X, filters=W1, stride=1, padding='SAME')
a1 = tf.nn.bias_add(conv1, b1)
z1 = tf.nn.relu(a1)
# Second convolutional layer
conv2 = tf.nn.conv1d(value=X, filters=W2, stride=1, padding='SAME')
a2 = tf.nn.bias_add(conv2, b2)
z2 = tf.nn.relu(a2)
# Third convolutional layer
conv3 = tf.nn.conv1d(value=X, filters=W3, stride=1, padding='SAME')
a3 = tf.nn.bias_add(conv3, b3)
z3 = tf.nn.relu(a3)

y_ = tf.nn.softmax(z3)

# Mask out the NoSeq
mask = tf.not_equal(tf.argmax(y, 2), 3)

y_masked = tf.boolean_mask(y, mask)
z3_masked = tf.boolean_mask(z3, mask)
y__masked = tf.boolean_mask(y_, mask)

# Define the loss function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_masked, logits=z3_masked))

# Define the optimizer operation
learning_rate = tf.placeholder(tf.float32)

#optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)

# Variables for prediction and accuracy
prediction = tf.argmax(y__masked, 1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, tf.argmax(y_masked, 1)), tf.float32))

# Initialize the variables (they are assigned default values)
init = tf.global_variables_initializer()

n_parameters = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])
print("Number of parameters:", n_parameters)


# Start as session
with tf.Session() as session:

    batch_size = 100
    # Run the initializer
    session.run(init)

    # Training cycle
    for epoch in range(10):
        print("Epoch:", epoch)
        for b in range(0, X_train.shape[0], batch_size):
            _, loss_value = session.run([optimizer, loss], feed_dict={X: X_train[b:b+batch_size],
                                                                      y: y_train[b:b+batch_size],
                                                                      learning_rate: 0.0001})
            if b % 1000 == 0:
                validation_accuracy = session.run(accuracy, feed_dict={X: X_validation, y: y_validation})
                print("loss[b=%04d] = %f, val_acc = %f" % (b, loss_value, validation_accuracy))

    print("Optimization done")

    # Calculate training accuracy
    train_accuracy_value, pred_train = session.run([accuracy, prediction], feed_dict={X: X_train, y: y_train})
    print("Train accuracy:", train_accuracy_value)

    # Calculate test accuracy
    test_accuracy_value, pred_test = session.run([accuracy, prediction], feed_dict={X: X_test, y: y_test})
    print("Test accuracy:", test_accuracy_value)

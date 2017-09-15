# autoencoder that receives a one-hot string of length 8
# e.g. [0, 0, 0, 1, 0, 0, 0, 0] with the goal to reproduce
# that same string as output

# TODO: find a way to access the activations of the the hidden layer

import tensorflow as tf
import numpy as np

# parameters
num_epochs = 50000

# network parameters
input_nodes = 8
hidden_nodes = 3
output_nodes = 8

# graph input
data = tf.placeholder("float", [None, input_nodes])

# create model
def model(x):
    hidden = tf.layers.dense(inputs=x, units=hidden_nodes, activation=tf.sigmoid)
    output = tf.layers.dense(inputs=hidden, units=output_nodes, activation=tf.sigmoid)

    return output

prediction = model(data)

# define loss and optimizer
loss_op = tf.reduce_mean(tf.losses.log_loss(labels=data, predictions=prediction))

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)

# evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# initialize variables
init = tf.global_variables_initializer()

# training
with tf.Session() as sess:
    sess.run(init)

    input_data = np.identity(8)

    for epoch in range(1, num_epochs):
        sess.run(train_op, feed_dict={data: input_data})

        loss, acc = sess.run([loss_op, accuracy], feed_dict={data: input_data})

        print("Epoch:", epoch, "| loss: ", loss, "| accuracy: ", acc)

    print("Optimization finished. Input recreations: ")
    print(prediction.eval(feed_dict= {data: input_data}))

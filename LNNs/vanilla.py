import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# parameters
num_epochs = 50000
batch_size = 128
print_every = 100 # print loss every 100 epochs

# variables
x = tf.placeholder("float", [None, 28*28])
y_ = tf.placeholder(tf.float32, [None, 10])

# create model
def model(X):
    # al4 = tf.layers.dense(inputs=X, units=784, activation=tf.sigmoid)

    input_layer = tf.reshape(X, [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=input_layer,
                            filters=64,
                            kernel_size=[3,3],
                            padding="same",
                            activation=tf.nn.relu)

    conv2 = tf.layers.conv2d(inputs=conv1,
                            filters=64,
                            kernel_size=[3,3],
                            padding="same",
                            activation=tf.nn.relu)

    conv3 = tf.layers.conv2d(inputs=conv2,
                            filters=3,
                            kernel_size=[3,3],
                            padding="same",
                            activation=tf.nn.relu)


    # flattened = tf.reshape(conv3, [-1, 4*4*3])
    flattened = tf.contrib.layers.flatten(conv3)
    dense = tf.layers.dense(inputs=flattened,
                            units=10,
                            activation=tf.nn.relu)



    return dense

output = model(x)
pred_classes = tf.argmax(input=output, axis=1)
pred_probs = tf.nn.softmax(output)



# define loss and optimizer
loss_op = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=output)

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)

# evaluate model


# initialize variables
init = tf.global_variables_initializer()

# training
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(1, num_epochs):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        writer = tf.summary.FileWriter('logs', sess.graph)
        # print(sess.run(output))
        sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
        writer.close()
        if epoch % print_every == 0:
            loss = sess.run(loss_op, feed_dict={x: batch_xs, y_: batch_ys})

            print("Epoch:", epoch, "| loss:", loss)

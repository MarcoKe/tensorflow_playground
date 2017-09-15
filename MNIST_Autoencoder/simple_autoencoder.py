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
num_reconstr_digits = 10 # how many reconstructed digits to display at the end

# network parameters
input_nodes = 28*28 # each image in the dataset has a size of 28*28px
hidden_nodes = 20 #input_nodes - 500 # completely arbitrary
output_nodes = input_nodes

# variables
x = tf.placeholder("float", [None, 28*28])

# create model
def model(X):
    encoded = tf.layers.dense(inputs=X, units=hidden_nodes, activation=tf.sigmoid)
    decoded = tf.layers.dense(inputs=encoded, units=output_nodes, activation=tf.sigmoid)

    return encoded, decoded

encoded, decoded = model(x)

# define loss and optimizer
loss_op = tf.reduce_mean(tf.losses.log_loss(labels=x, predictions=decoded))

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
        sess.run(train_op, feed_dict={x: batch_xs})

        if epoch % print_every == 0:
            loss = sess.run(loss_op, feed_dict={x: batch_xs})
            print("Epoch:", epoch, "| loss:", loss)

    # decoded images on test data:
    test_mnist = {x: mnist.test.images}
    decoded_images = decoded.eval(test_mnist)
    print("Loss on test set:", loss_op.eval(test_mnist))

plt.figure()
original_images = mnist.test.images

for i in range(num_reconstr_digits):
    # original
    ax = plt.subplot(2, num_reconstr_digits, i+1)
    plt.imshow(original_images[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # reconstructed
    ax = plt.subplot(2, num_reconstr_digits, i+1+num_reconstr_digits)
    plt.imshow(decoded_images[i].reshape(28,28))
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.savefig('mnist_simple_autoencoder.png')

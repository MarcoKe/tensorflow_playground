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
F1 =  np.repeat(np.array([[0] *7* i + [1]*7 + [0] * 7 * np.abs(i-((int(784/7))-1)) for i in range(int(784/7))]), 7, axis=0)
F2 =  np.ones((784, 392))
F3 =  np.repeat(np.array([[0] *7* i + [1]*7 + [0] * 7 * np.abs(i-((int(392/7))-1)) for i in range(int(392/7))]), 7, axis=0)
F4 =  np.ones((392, 196))
F5 =  np.repeat(np.array([[0] *7* i + [1]*7 + [0] * 7 * np.abs(i-((int(196/7))-1)) for i in range(int(196/7))]), 7, axis=0)
F6 =  np.ones((196, 98))
F7 =  np.repeat(np.array([[0] *7* i + [1]*7 + [0] * 7 * np.abs(i-((int(98/7))-1)) for i in range(int(98/7))]), 7, axis=0)
F8 =  np.ones((98, 49))
F9 =  np.repeat(np.array([[0] *7* i + [1]*7 + [0] * 7 * np.abs(i-((int(49/7))-1)) for i in range(int(49/7))]), 7, axis=0)
F10 =  np.ones((49, 7))



x = tf.placeholder("float", [None, 28*28])
y_ = tf.placeholder(tf.float32, [None, 10])

tfF1 = tf.constant(F1.astype(np.float32), shape=[784, 784])
tfF2 = tf.constant(F2.astype(np.float32), shape=[784, 392])
tfF3 = tf.constant(F3.astype(np.float32), shape=[392, 392])
tfF4 = tf.constant(F4.astype(np.float32), shape=[392, 196])
tfF5 = tf.constant(F5.astype(np.float32), shape=[196, 196])
tfF6 = tf.constant(F6.astype(np.float32), shape=[196, 98])
tfF7 = tf.constant(F7.astype(np.float32), shape=[98, 98])
tfF8 = tf.constant(F8.astype(np.float32), shape=[98, 49])
tfF9 = tf.constant(F9.astype(np.float32), shape=[49, 49])
tfF10 = tf.constant(F10.astype(np.float32), shape=[49, 7])


# input = tflearn.input_data(shape=[None, num_input])
# tf_F = tf.constant(F, shape=[20, 10])
#
# first = tflearn.fully_connected(input, 20, activation='relu')
# # Here is where I want to use a custom function, that uses my F matrix
# # I want only connections that are ones (and not zeros) in F
#
# W = tf.Variable(tf.random_uniform([20, 10]), name='Weights')
# b = tf.Variable(tf.zeros([10]), name='biases')
# W_filtered = tf.multiply(tf_F, W)
# second = tf.matmul( W_filtered, first) + b

# create model
def model(X):
    # al4 = tf.layers.dense(inputs=X, units=784, activation=tf.sigmoid)

    w1 = tf.Variable(tf.random_uniform([784, 784]), name="w-AL4-AL3")
    b1 = tf.Variable(tf.zeros([784]), name="b-AL4-AL3")
    w1 = tf.multiply(tfF1, w1)

    al3 = tf.nn.relu(tf.add(tf.matmul(X, w1, name='AL3-mult'), b1, name='AL3-add'))

    # al3 = tf.nn.relu(al3)

    w2 = tf.Variable(tf.random_uniform([784, 392]), name="w-AL3-BL4")
    b2 = tf.Variable(tf.zeros([392]), name="b-AL3-BL4")
    w2 = tf.multiply(tfF2, w2)

    bl4 = tf.nn.relu(tf.add(tf.matmul(al3, w2, name='BL4-mult'), b2, name='BL4-add'))

    # bl4 = tf.nn.relu(bl4)

    w3 = tf.Variable(tf.random_uniform([392, 392]), name="w-BL4-BL3")
    b3 = tf.Variable(tf.zeros([392]), name="b-BL4-BL3")
    # w3 = tf.multiply(tfF3, w3)

    bl3 = tf.nn.relu(tf.add(tf.matmul(bl4, w3, name='BL3-mult'), b3, name='BL3-add'))

    # bl3 = tf.nn.relu(bl3)

    w4 = tf.Variable(tf.random_uniform([392, 196]), name="w-BL3-CL4")
    b4 = tf.Variable(tf.zeros([196]), name="b-BL3-CL4")
    # w4 = tf.multiply(tfF4, w4)

    cl4 = tf.nn.relu(tf.add(tf.matmul(bl3, w4, name='CL4-mult'), b4, name='CL4-add'))

    # cl4 = tf.nn.relu(cl4)

    w5 = tf.Variable(tf.random_uniform([196, 196]), name="w-CL4-CL3")
    b5 = tf.Variable(tf.zeros([196]), name="b-CL4-CL3")
    # w5 = tf.multiply(tfF5, w5)

    cl3 = tf.nn.relu(tf.add(tf.matmul(cl4, w5, name='CL3-mult'), b5, name='CL3-add'))

    # cl3 = tf.nn.relu(cl3)

    w6 = tf.Variable(tf.random_uniform([196, 98]), name="w-CL3-DL4")
    b6 = tf.Variable(tf.zeros([98]), name="b-CL3-DL4")
    # w6 = tf.multiply(tfF6, w6)

    dl4 = tf.nn.relu(tf.add(tf.matmul(cl3, w6, name='DL4-mult'), b6, name='DL4-add'))

    # dl4 = tf.nn.relu(dl4)

    w7 = tf.Variable(tf.random_uniform([98, 98]), name="w-DL4-DL3")
    b7 = tf.Variable(tf.zeros([98]), name="b-DL4-DL3")
    # w7 = tf.multiply(tfF7, w7)

    dl3 = tf.nn.relu(tf.add(tf.matmul(dl4, w7, name='DL3-mult'), b7, name='DL3-add'))

    # dl3 = tf.nn.relu(dl3)

    w8 = tf.Variable(tf.random_uniform([98, 49]), name="w-DL3-EL4")
    b8 = tf.Variable(tf.zeros([49]), name="b-DL3-EL4")
    # w8 = tf.multiply(tfF8, w8)

    el4 = tf.nn.relu(tf.add(tf.matmul(dl3, w8, name='EL4-mult'), b8, name='EL4-add'))

    # el4 = tf.nn.relu(el4)

    w9 = tf.Variable(tf.random_uniform([49, 49]), name="w-EL4-EL3")
    b9 = tf.Variable(tf.zeros([49]), name="b-EL4-EL3")
    # w9 = tf.multiply(tfF9, w9)

    el3 = tf.nn.relu(tf.add(tf.matmul(el4, w9, name='EL3-mult'), b9, name='EL3-add'))


    w10 = tf.Variable(tf.random_uniform([49, 10]), name="w-EL3-FL4")
    b10 = tf.Variable(tf.zeros([10]), name="b-EL3-FL4")
    # # w10 = tf.multiply(tfF10, w10)

    fl4 = tf.add(tf.matmul(el3, w10, name='FL4-mult'), b10, name='FL4-add')

    y = tf.nn.softmax(fl4)


    return y

output = model(x)

# define loss and optimizer
loss_op = tf.reduce_mean(tf.losses.log_loss(labels=y_, predictions=output))
# loss_op = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(output), reduction_indices=[1]))

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

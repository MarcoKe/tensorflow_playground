import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# parameters
save_path = "model.ckp"
num_steps = 1000
batch_size = 128
print_every = 100 # print loss every 100 epochs
num_reconstr_digits = 10 # how many reconstructed digits to display at the end

# network parameters
input_nodes = 28*28 # each image in the dataset has a size of 28*28px
encode1_nodes = 392 #input_nodes - 500 # completely arbitrary
encode2_nodes = 196
decode1_nodes = encode1_nodes
output_nodes = input_nodes

# variables
x = tf.placeholder("float", [None, 28*28])

def gain(x):
    sat_thresh = 20
    #return 1/(1-tf.cond(x < sat_thresh, lambda: (2/3)/sat_thresh*x, lambda: 2/3))
    # tmp = tf.where(x < sat_thresh, ((2/3)/sat_thresh)*x, tf.ones(x.shape)*(2/3))
    tmp = tf.minimum((2/3)/sat_thresh * x, 2/3)
    return 1/(1-tmp)
    # return 1/(1-tf.case({tf.less(x, sat_thresh): lambda: (2/3)/sat_thresh*x, tf.greater_equal(x, sat_thresh): lambda: 2/3}, default= lambda: 2/3))
    # return 1/(1-np.piecewise(x, [x < sat_thresh, x >= sat_thresh], [lambda x: (2/3)/sat_thresh*x, lambda x: 2/3]))


with tf.name_scope('encoder1_t1') as scope:
        w1 = tf.Variable(tf.truncated_normal([784, encode1_nodes], stddev=0.0001), name="w-input-encoder1")
        b1 = tf.Variable(tf.constant(0.1, shape=[encode1_nodes]), name="b-input-encoder1")
        encoder1 = tf.nn.relu(tf.add(tf.matmul(x, w1, name="mul1"), b1))

    # (, 392) x (392, 196) = (, 196)
with tf.name_scope('encoder2_t1') as scope:
    w2 = tf.Variable(tf.truncated_normal([encode1_nodes, encode2_nodes], stddev=0.0001), name="w-encoder1-encoder2")
    b2 = tf.Variable(tf.constant(0.1, shape=[encode2_nodes]), name="b-encoder1-encoder2")
    encoder2 = tf.nn.relu(tf.add(tf.matmul(encoder1, w2, name="mul2"), b2))

# (, 196) x (196, 392) = (, 392)
with tf.name_scope('decoder1_t1') as scope:
    w3 = tf.Variable(tf.truncated_normal([encode2_nodes, decode1_nodes], stddev=0.0001), name="w-encoder2-decoder1")
    b3 = tf.Variable(tf.constant(0.1, shape=[decode1_nodes]), name="b-encoder2-decoder1")
    decoder1 = tf.nn.relu(tf.add(tf.matmul(encoder2, w3, name="mul3"), b3))

# (, 392) x (392, 784) = (, 784)
with tf.name_scope('decoder2_t1') as scope:
    w4 = tf.Variable(tf.truncated_normal([decode1_nodes, output_nodes], stddev=0.0001), name="w-decoder1-decoder2")
    b4 = tf.Variable(tf.constant(0.1, shape=[output_nodes]), name="b-decoder1-decoder2")
    decoder2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder1, w4, name="mul4"), b4))

##
with tf.name_scope('encoder1_t2') as scope:
    encoder1_2 = tf.add(tf.matmul((x), w1, name="mul5"), b1)
    encoder1_2 = 1/(1-tf.minimum((2/3)/20 * decoder1, 2/3)) * tf.maximum(0.0, encoder1_2)


with tf.name_scope('encoder2_t2') as scope:
    encoder2_2 = ncoder2_2 = tf.nn.relu(tf.add(tf.matmul((encoder1_2), w2, name="mul6"), b2))

with tf.name_scope('decoder1_t2') as scope:
    decoder1_2 = tf.nn.relu(tf.add(tf.matmul(encoder2_2, w3), b3))

with tf.name_scope('decoder2_t2') as scope:
    decoder2_2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder1_2, w4), b4))



    
decoded = decoder2_2

# define loss and optimizer
# loss_op = tf.reduce_mean(tf.losses.log_loss(labels=x, predictions=decoded)+tf.losses.log_loss(labels=x, predictions=decoder2))
loss_op = tf.reduce_mean(tf.losses.log_loss(labels=x, predictions=decoded))


optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)
# check_op = tf.add_check_numerics_ops()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# initialize variables
init = tf.global_variables_initializer()

# training
sess = tf.Session()
sess.run(init)

for step in range(1, num_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    writer = tf.summary.FileWriter('logs', sess.graph)
    # sess.run(check_op, feed_dict={x: batch_xs})
    sess.run(train_op, feed_dict={x: batch_xs})
    writer.close()

    if step % print_every == 0:
        loss = sess.run(loss_op, feed_dict={x: batch_xs})
        print("Step:", step, "| Minibatch loss:", loss)
        with sess.as_default():
            test_mnist = {x: mnist.test.images}
            print("Loss on test set:", loss_op.eval(test_mnist))

# decoded images on test data:
with sess.as_default():
    test_mnist = {x: mnist.test.images}
    decoded_images = decoded.eval(test_mnist)
    print("Loss on test set:", loss_op.eval(test_mnist))

    save_path = saver.save(sess, save_path)
    print("Parameters saved as:" + save_path)
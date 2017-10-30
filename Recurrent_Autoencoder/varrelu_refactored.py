import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# parameters
save_parameters = True
load_parameters = False
save_path = "/tmp/model.ckp"
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

def create_weights(dimensions, name):
    return tf.Variable(tf.truncated_normal(dimensions, stddev=0.0001), name=name)

def create_bias(dimensions, name):
    return tf.Variable(tf.zeros(dimensions), name=name)

w1 = create_weights([784, encode1_nodes], "w-input_encoder1")
b1 = create_bias([encode1_nodes], "b-input-encoder1")
w2 = create_bias([encode1_nodes, encode2_nodes], "w-encoder1-encoder2")
b2 = create_bias([encode2_nodes], "b-encoder1-encoder2")
w3 = create_weights([encode2_nodes, decode1_nodes], "w-encoder2-decoder1")
b3 = create_bias([decode1_nodes], "b-encoder2-decoder1")
w4 = create_weights([decode1_nodes, output_nodes], "w-decoder1-decoder2")
b4 = create_bias([output_nodes], "b-decoder1-decoder2")

# create model
def model(X):
    # (, 784) x (784, 392) = (, 392)
    with tf.name_scope('encoder1_t1') as scope:
        encoder1 = tf.nn.relu(tf.add(tf.matmul(X, w1, name="mul1"), b1))

    # (, 392) x (392, 196) = (, 196)
    with tf.name_scope('encoder2_t1') as scope:
        encoder2 = tf.nn.relu(tf.add(tf.matmul(encoder1, w2, name="mul2"), b2))

    # (, 196) x (196, 392) = (, 392)
    with tf.name_scope('decoder1_t1') as scope:
        decoder1 = tf.nn.relu(tf.add(tf.matmul(encoder2, w3, name="mul3"), b3))

    # (, 392) x (392, 784) = (, 784)
    with tf.name_scope('decoder2_t1') as scope:
        decoder2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder1, w4, name="mul4"), b4))

    # second pass: 

    with tf.name_scope('encoder1_t2') as scope:
        encoder1_2 = tf.add(tf.matmul((X), w1, name="mul5"), b1)
        encoder1_2 = 1/(1-tf.minimum((2/3)/20 * decoder1, 2/3)) * tf.maximum(0.0, encoder1_2)

    with tf.name_scope('encoder2_t2') as scope:
        encoder2_2 = ncoder2_2 = tf.nn.relu(tf.add(tf.matmul((encoder1_2), w2, name="mul6"), b2))
    
    with tf.name_scope('decoder1_t2') as scope:
        decoder1_2 = tf.nn.relu(tf.add(tf.matmul(encoder2_2, w3), b3))

    with tf.name_scope('decoder2_t2') as scope:
        decoder2_2 = tf.nn.sigmoid(tf.add(tf.matmul(decoder1_2, w4), b4))

    return decoder2, decoder2_2

decoded_first_pass, decoded = model(x)

# define loss and optimizer
loss_op = tf.reduce_mean(tf.losses.log_loss(labels=x, predictions=decoded)+tf.losses.log_loss(labels=x, predictions=decoded_first_pass))

optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)
# check_op = tf.add_check_numerics_ops()

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# add summaries 
loss_summary = tf.summary.scalar("training_loss", loss_op)


# initialize variables
init = tf.global_variables_initializer()

# training
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter("logs", sess.graph)

    if load_parameters: 
        saver.restore(sess, save_path)
        print("Parameters restored.")

    for step in range(1, num_steps):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        writer = tf.summary.FileWriter('logs', sess.graph)
        # sess.run(check_op, feed_dict={x: batch_xs})
        sess.run(train_op, feed_dict={x: batch_xs})
        writer.close()

        if step % print_every == 0:
            loss, loss_summ = sess.run([loss_op, loss_summary], feed_dict={x: batch_xs})
            summary_writer.add_summary(loss_summ, step) 
            summary_writer.flush()
            print("Step:", step, "| Minibatch loss:", loss)

    # decoded images on test data:
    test_mnist = {x: mnist.test.images}
    decoded_images = decoded.eval(test_mnist)
    print("Loss on test set:", loss_op.eval(test_mnist))

    if save_parameters:
        save_path = saver.save(sess, save_path)
        print("Parameters saved as:" + save_path)

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

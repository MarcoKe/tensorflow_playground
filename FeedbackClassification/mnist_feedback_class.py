import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# import data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("../MNIST_data/", one_hot=True)

# parameters
save_path = "./model.ckp"
num_steps = 10000
batch_size = 128
print_every = 100 # print loss every 100 epochs
num_reconstr_digits = 10 # how many reconstructed digits to display at the end

# network parameters
input_nodes = 28*28 # each image in the dataset has a size of 28*28px
h1_nodes = 392 #input_nodes - 500 # completely arbitrary
h2_nodes = 196
output_nodes = 10

# variables
x = tf.placeholder("float", [None, 28*28])
y_ = tf.placeholder(tf.float32, [None, 10])

with tf.name_scope('h1_t1') as scope:
    w1 = tf.Variable(tf.truncated_normal([784, h1_nodes], stddev=0.0001), name="w-input-encoder1")
    b1 = tf.Variable(tf.constant(0.1, shape=[h1_nodes]), name="b-input-encoder1")
    h1 = tf.nn.relu(tf.add(tf.matmul(x, w1, name="mul1"), b1))

    # (, 392) x (392, 196) = (, 196)
with tf.name_scope('h2_t1') as scope:
    w2 = tf.Variable(tf.truncated_normal([h1_nodes, h2_nodes], stddev=0.0001), name="w-encoder1-encoder2")
    b2 = tf.Variable(tf.constant(0.1, shape=[h2_nodes]), name="b-encoder1-encoder2")
    h2 = tf.nn.relu(tf.add(tf.matmul(h1, w2, name="mul2"), b2))


with tf.name_scope('o_t1') as scope:
    w3 = tf.Variable(tf.truncated_normal([h2_nodes, output_nodes], stddev=0.0001), name="w-encoder2-decoder1")
    b3 = tf.Variable(tf.constant(0.1, shape=[output_nodes]), name="b-encoder2-decoder1")
    output_1 = tf.nn.relu(tf.add(tf.matmul(h2, w3, name="mul3"), b3))

with tf.name_scope('h1_t2') as scope: 
    w4 = tf.Variable(tf.truncated_normal([h2_nodes+output_nodes, h1_nodes], stddev=0.0001))
    h1_2 = tf.add(tf.matmul(x, w1, name="mul4"), b1)
    fb = tf.matmul(tf.concat([h2, output], 1), w4)
    h1_2 = 1/(1-tf.minimum((2/3)/20 * fb, 2/3)) * tf.maximum(0.0, h1_2)

with tf.name_scope('h2_t2') as scope: 
    w5 = tf.Variable(tf.truncated_normal([output_nodes, h2_nodes], stddev=0.0001))
    h2_2 = tf.add(tf.matmul(h1, w2, name="mul2"), b2)
    fb = tf.matmul(output, w5)
    h2_2 = 1/(1-tf.minimum((2/3)/20 * fb, 2/3)) * tf.maximum(0.0, h2_2)

with tf.name_scope('o_t2') as scope: 
    output_2 = tf.nn.relu(tf.add(tf.matmul(h2_2, w3), b3))




 output = output_2   

pred_classes = tf.argmax(input=output, axis=1)
pred_probs = tf.nn.softmax(output)



# define loss and optimizer
loss_op = tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=output)


optimizer = tf.train.AdamOptimizer()
train_op = optimizer.minimize(loss_op)
# check_op = tf.add_check_numerics_ops()

# evaluate model
correct_pred = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

# initialize variables
init = tf.global_variables_initializer()

# training
sess = tf.Session()
sess.run(init)

for step in range(1, num_steps):
    batch_xs, batch_ys = mnist.train.next_batch(batch_size)
    #writer = tf.summary.FileWriter('logs', sess.graph)
    # sess.run(check_op, feed_dict={x: batch_xs})
    sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
    #writer.close()

    if step % print_every == 0:
        loss, acc = sess.run([loss_op, accuracy], feed_dict={x: batch_xs, y_: batch_ys})

        print("Step:", step, "| loss:", loss, "| accuracy:", acc)

        with sess.as_default():
            test_mnist = {x: mnist.test.images, y_: mnist.test.labels}
            print("Loss on test set:", loss_op.eval(test_mnist), "Accuracy: accuracy:", accuracy.eval(test_mnist))



        


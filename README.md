# tensorflow_playground

Setting up the environment: 
==============================

- clone the project and change into the project directory 
- run `conda env create` 
- run `source activate tfenv` 
- run `pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0-py3-none-any.whl`

- test the environment by entering the python interactive shell and executing: 
```import tensorflow as tf
	hello = tf.constant('Hello, TensorFlow!')
	sess = tf.Session()
	print(sess.run(hello))```
  
 

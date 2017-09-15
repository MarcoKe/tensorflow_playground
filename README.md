# tensorflow_playground

## Setting up the environment (Mac): 


- clone the project and change into the project directory 
- run `conda env create` 
- run `source activate tfenv` 
- run `pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0-py3-none-any.whl`

- test the environment by entering the python interactive shell and executing: 
```
	import tensorflow as tf
	hello = tf.constant('Hello, TensorFlow!')
	sess = tf.Session()
	print(sess.run(hello))
```
- might get some warnings in the form “The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.”, the installation will work properly, but speedups could be achieved by compiling tensorflow from source
 

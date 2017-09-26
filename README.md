# tensorflow_playground

## Setting up the environment (Mac): 
- *(this is my preferred way of doing it, for alternatives look [here](https://www.tensorflow.org/install/)).*

- clone the project and change into the project directory 
- run `conda env create` (in a terminal)
- run `source activate tfenv` 
- run `pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.3.0-py3-none-any.whl`

- test the environment by entering the python interactive shell and executing: 
```python
	import tensorflow as tf
	hello = tf.constant('Hello, TensorFlow!')
	sess = tf.Session()
	print(sess.run(hello))
```
- might get some warnings in the form “The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.”, the installation will work properly, but speedups could be achieved by compiling tensorflow from source

## (Windows) 

- clone the project and change into the project directory 
- run `conda env create -f environment.yml` (in the Anaconda prompt)
- run `activate tfenv`
- run `pip install --ignore-installed --upgrade tensorflow`

- then proceed to test the environment as described above
 

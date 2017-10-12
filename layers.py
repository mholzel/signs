import random
import numpy as np
import tensorflow as tf
from time import sleep


class Layer:
    def __init__(self, operation=lambda x: x):
        self.operation = operation

    def output(self, input):
        return self.operation(input)

    def L1(self):
        return tf.constant(0.0)

    def L2(self):
        return tf.constant(0.0)


class FlattenLayer(Layer):
    def __init__(self):
        super().__init__(lambda x: tf.contrib.layers.flatten(x))


class DropoutLayer(Layer):
    def __init__(self, keep_prob=1):
        super().__init__(lambda x: tf.nn.dropout(x, keep_prob=keep_prob))


class WeightedLayer(Layer):
    def __init__(self, weight_shape, name="weighted_layer", mean=0, stddev=0.1):
        self.W = tf.Variable(tf.truncated_normal(shape=weight_shape, mean=mean, stddev=stddev), name=name + "_weights")
        self.b = tf.Variable(tf.zeros(weight_shape[-1]), name=name + "_bias")
        # Subclasses should override self.operation after calling super().__init__

    def L1(self):
        return tf.norm(self.W, ord=1) + tf.norm(self.b, ord=1)

    def L2(self):
        return tf.sqrt(tf.norm(self.W) ** 2 + tf.norm(self.b) ** 2)


class FullLayer(WeightedLayer):
    def __init__(self, weight_shape, name="full_layer", mean=0, stddev=0.1, activation=tf.nn.relu):
        super().__init__(weight_shape=weight_shape, name=name, mean=mean, stddev=stddev)
        self.operation = lambda x: activation(tf.matmul(x, self.W) + self.b)


class ConvLayer(WeightedLayer):
    def __init__(self, weight_shape=(5, 5, 1, 6), name="conv_layer", mean=0, stddev=0.1, pool_size=[1, 2, 2, 1],
                 pool_stride=[1, 2, 2, 1],
                 activation=tf.nn.relu):
        super().__init__(weight_shape=weight_shape, name=name, mean=mean, stddev=stddev)
        convolver = lambda x: tf.nn.conv2d(x, self.W, strides=[1, 1, 1, 1], padding='VALID') + self.b
        pooler = lambda y: tf.nn.max_pool(y, ksize=pool_size, strides=pool_stride, padding='VALID')
        self.operation = lambda z: pooler(activation(convolver(z)))


class LeNet:
    def __init__(self, x, y, depths=[10, 20, 200, 100]):
        '''
        Create a LeNet with a 4-dimensional input with the shape:

        (# of examples, image width, image height, image channels)

        If the image channels = 1 (such as in greyscale images), and you specify a
        3-dimensional input, then this function will add the extra dimension for you.

        Unlike the traditional LeNet, you can create this type of network for input images of any size.
        Furthermore, all of the layer dimensions are configurable by changing the "depths" list.
        Specifically, the LeNet consists of 8 layers:

        1) A 2d convolution of size (5, 5, channels, depths[0]), where channels denotes the number of input channels.
        For instance, typically you will have that channels = 1 (greyscale) or channels = 3 (rgb)
        2) A 2d convolution of size (5, 5, depths[0], depths[1])
        3) A vectorization of the output of the previous layer
        4) A fully-connected layer (with ReLu activations) of size (n, depths[2]), where n denotes the size of the output of the previous layer
        5) A dropout layer
        6) A fully-connected layer (with ReLu activations) of size (depths[2], depths[3])
        7) A dropout layer
        8) A fully-connected layer (with unity activations) of size (depths[3], p), where p denotes the number of classes in y

        :param x: A tf.placeholder for the input images (the first dimension should be specified as None)
        :param y: A tf.placeholder for one-hot encoded annotations (the first dimension should be specified as None)
        :param depths: Used to configure the neural net hidden layer sizes as described above
        '''
        self.x = x
        self.y = y
        x_shape = self.x.get_shape().as_list()
        if len(x_shape) == 3:
            x_shape.append(1)
            # Replace "None" values with -1
            x_shape = [s if s != None else -1 for s in x_shape]
            z = tf.reshape(self.x, x_shape)
        else:
            z = self.x

        # The dropout placeholder which will control the dropout rate
        self.dropout_keep_prob = tf.placeholder(tf.float32)

        # We store each of the internal layers and outputs
        self.layers = []
        self.output = [z]

        # A convenience method for adding an internal layer
        def add(L):
            self.layers.append(L)
            self.output.append(L.output(self.output[-1]))

        # Create all of the layers
        add(ConvLayer(name="layer_1", weight_shape=(5, 5, x_shape[-1], depths[0])))
        add(ConvLayer(name="layer_2", weight_shape=(5, 5, depths[0], depths[1])))
        add(FlattenLayer())

        n = self.output[-1].get_shape().as_list()[-1]

        add(FullLayer(name="layer_3", weight_shape=(n, depths[2])))
        add(DropoutLayer(keep_prob=self.dropout_keep_prob))
        add(FullLayer(name="layer_4", weight_shape=(depths[2], depths[3])))
        add(DropoutLayer(keep_prob=self.dropout_keep_prob))
        add(FullLayer(name="layer_5", weight_shape=(depths[3], self.y.get_shape().as_list()[1]),
                      activation=lambda q: q))

        # Give names to a bunch of values that we will use during evaluation
        self.logits = self.output[-1]
        self.predictions = tf.argmax(self.logits, axis=1)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(self.y, axis=1)), tf.float32))
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y)
        self.cost = tf.reduce_mean(self.cross_entropy)

    def L1(self):
        norm = tf.constant(0.0)
        for layer in self.layers:
            norm += layer.L1()
        return norm

    def L2(self):
        norm = tf.constant(0.0)
        for layer in self.layers:
            norm += layer.L2()
        return norm

    def stats(self):
        '''
        :return: The objects used to evaluate this network. To use these, you should
        "run" this function in a session, and pass the output to the "display" function:

        lenet = LeNet(...)
        with tf.Session as s:
            lenet.display( s.run( lenet.stats(), feed_dict=... ) )
        '''
        return self.cost, self.logits, self.predictions, tf.argmax(self.y, axis=1), self.accuracy

    def display(self, stats, name="Training  "):
        '''
        Display the statistics output by the "stats" function. To use these, you should "run"
        this function in a session, and pass the output to the "display" function:

        lenet = LeNet(...)
        with tf.Session as s:
            lenet.display( s.run( lenet.stats(), feed_dict=... ) )
        :param stats:
        :return:
        '''
        costs, logits, predictions, y, accuracy = stats
        print(name + " (Accuracy,Cost) : (", accuracy, ", ", costs, ")")

    def train(self, X_train, y_train, trainer, load_from=None, save_to=None,
              dropout_keep_prob=0.3, X_valid=None, y_valid=None, 
              X_test=None, y_test=None, epochs=20, batch_size=1024):
        '''
        Train the network using the specified data. The validation data is optional,
        but helpful for diagnosing whether to stop the training due to overfitting.
        '''

        # Set up the feed dictionarires
        train_dict = {self.x: X_train, self.y: y_train, self.dropout_keep_prob: 1}
        valid_dict = {self.x: X_valid, self.y: y_valid, self.dropout_keep_prob: 1}
        test_dict  = {self.x: X_test,  self.y:  y_test, self.dropout_keep_prob: 1}

        # Create a dictionary of the variables.
        # We will inspect how these change at every training epoch.
        # If they stop changing, then the optimization has converged.
        vars = {}

        # Class used to save and/or restore Tensor Variables
        saver = tf.train.Saver(var_list=tf.trainable_variables())

        # Finally, loop. We create a list of indices that we will repeatedly shuffle
        N = len(y_train)
        ind = list(range(N))
        with tf.Session() as sess:

            # Make sure that all of the variables are initialized, loading from file if possible
            sess.run(tf.global_variables_initializer())
            if load_from is not None:
                try:
                    saver.restore(sess, load_from)
                    print("Loaded from " + load_from)
                except:
                    print("Could not load from " + load_from)

            # Now train for the specified number of epochs
            self.display(sess.run(self.stats(), feed_dict=train_dict), name="Initial   ")
            for i in range(epochs):

                random.shuffle(ind)
                for offset in range(0, N, batch_size):
                    end = min(offset + batch_size, N)
                    feed_dict = {self.x: X_train[ind[offset:end]], self.y: y_train[ind[offset:end]],
                                 self.dropout_keep_prob: dropout_keep_prob}
                    sess.run(trainer, feed_dict=feed_dict)

                print("Epoch: " + str(i + 1) + " / " + str(epochs))
                self.display(sess.run(self.stats(), feed_dict=train_dict))

                # Show how much the variables have changed
                change = 0
                for v in tf.global_variables():
                    tmp = sess.run(v)
                    if v.name in vars:
                        # Let's see how much this changed:
                        change += np.linalg.norm(vars[v.name] - tmp) ** 2
                    vars[v.name] = tmp
                if change != 0:
                    print("Squared change in variables: ", change)

                # If you provided validation data, we can show you those statistics
                if X_valid is not None and y_valid is not None:
                    self.display(sess.run(self.stats(), feed_dict=valid_dict), name="Validation")

                # Save the variables if the validation accuracy improved
                if save_to is not None:
                    saver.save(sess, save_to, global_step=i)
                    
            # If you provided test data, we can show you those statistics
            if X_test is not None and y_test is not None:
                self.display(sess.run(self.stats(), feed_dict=test_dict), name="Testing   ")
            
            if X_valid is not None and y_valid is not None:
                softmax = tf.nn.softmax(self.logits)
                return sess.run(softmax, feed_dict=valid_dict)
            else:
                return None

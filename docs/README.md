# German traffic sign classification in Tensorflow
The goal of this project is to develop a modified `LeNet` architecture to classify German traffic signs in `Tensorflow` (for `Python`). Furthermore, we aim to make this code as general as possible so that it can easily be extended and adapted for other image processing projects and pipelines.

## Building blocks
Neural networks are typically discussed in terms of their layers, where each layer is responsible for two things:
1. Mapping input tensors to output tensors
1. Computing `L1` and `L2` norms of its weights (and biases).

> The reason we highlight the second aspect is that it is fairly common to use the `L1` or `L2` norm of a network's weights (and biases) as a regularization term in the cost function.

Hence the `layers` package defines the following base class:

    import tensorflow as tf
    class Layer:
        def __init__(self, operation=lambda x:x, name="layer"):
            self.name       = name
            self.operation  = operation

        def output(self, input):
            return self.operation(input)

        def L1(self):
            return tf.constant(0.0)

        def L2(self):
            return tf.constant(0.0)

A simple example of how you can extend this base `Layer` class is given by the `FlattenLayer` class, which (as the name implies) simply flattens the input tensor into a single 1D vector:

    class FlattenLayer(Layer):
        def __init__(self, name="flatten_layer"):
            super().__init__(lambda x: tf.contrib.layers.flatten(x), name=name)

An important thing to note about the `Layer` and `FlattenLayer` classes is that they do not define any internal weights or biases. Hence their `L1` and `L2` norms are 0. Conversely, many layers would more appropriately be represented as subclasses of the `WeightedLayer` class:

    class WeightedLayer(Layer):
        def __init__(self, weight_shape, name="weighted_layer", mean=0, stddev=0.1):
            self.W = tf.Variable(tf.truncated_normal(shape=weight_shape, mean=mean, stddev=stddev), name=name + "_weights")
            self.b = tf.Variable(tf.zeros(weight_shape[-1]), name=name + "_bias")
            # Subclasses should override self.operation after calling super().__init__

        def L1(self):
            return tf.norm(self.W, ord=1) + tf.norm(self.b, ord=1)

        def L2(self):
            return tf.sqrt(tf.norm(self.W) ** 2 + tf.norm(self.b) ** 2)


which defines weights and biases according to the specified `weight_shape`, and initializes the weights using the specified mean and standard deviation. For example, `FullLayer` is a subclass of `WeightedLayer` which also allows the user to specify an `activation` function:

    class FullLayer(WeightedLayer):
        def __init__(self, weight_shape, name="full_layer", mean=0, stddev=0.1, activation=tf.nn.relu):
            super().__init__(weight_shape=weight_shape, name=name, mean=mean, stddev=stddev)
            self.operation = lambda x: activation(tf.matmul(x, self.W) + self.b)

Similarly, a convolution layer is a `WeightedLayer` which chains together a convolution operation, activiation function, and max-pooling operation:

    class ConvLayer(WeightedLayer):
        def __init__(self, weight_shape=(5, 5, 1, 6), name="conv_layer", mean=0, stddev=0.1, pool_size=[1, 2, 2, 1], pool_stride=[1, 2, 2, 1], activation=tf.nn.relu):
            super().__init__(weight_shape=weight_shape, name=name, mean=mean, stddev=stddev)
            convolver      = lambda x: tf.nn.conv2d(x, self.W, strides=[1, 1, 1, 1], padding='VALID') + self.b
            pooler         = lambda y: tf.nn.max_pool(y, ksize=pool_size, strides=pool_stride, padding='VALID')
            self.operation = lambda z: pooler( activation( convolver( z )) )

From these basic building blocks, you should be able to easily and intuitively construct many neural networks found in the literature.

## LeNet
`LeNet` is one of the most famous convolutional neural networks, and can be represented as a chain of the building blocks described above. Specifically, the basic `LeNet` architecture looks like this:

    class LeNet:

        def __init__(self, x, y, depths=[10, 20, 200, 100]):
            '''
            :param x: A tf.placeholder for the input images of size

                    (None) X (image width) X (image height) X (image channels)

            :param y: A tf.placeholder for one-hot encoded annotations

                    (None) X (# of image classes)
            '''

            # Create all of the layers
            m = x.get_shape().as_list()[-1]

            x = ConvLayer(name="layer_1", weight_shape=(5, 5,     m    , depths[0])).output(x)
            x = ConvLayer(name="layer_2", weight_shape=(5, 5, depths[0], depths[1])).output(x)
            x = FlattenLayer().output(x)

            n = x.get_shape().as_list()[-1]

            x = FullLayer(name="layer_3", weight_shape=(    n    , depths[2])).output(x)
            x = FullLayer(name="layer_4", weight_shape=(depths[2], depths[3])).output(x)
            x = FullLayer(name="layer_5", weight_shape=(depths[3], y.get_shape().as_list()[1]), activation=lambda q: q).output(x)

            # Give names to a bunch of values that we will use during evaluation
            self.logits         = x
            self.y              = y
            self.predictions    = tf.argmax(self.logits, axis=1)
            self.accuracy       = tf.reduce_mean(tf.cast(tf.equal(self.predictions, tf.argmax(y, axis=1)), tf.float32))
            self.cross_entropy  = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=y)
            self.cost           = tf.reduce_mean(self.cross_entropy)

        ...

The beauty of defining `LeNet` like this is that we do not have to hardcode any of the problem-specific details. For instance, the traditional `LeNet` architecture was developed for the `MNIST` problem, meaning that it was designed to take in 32x32 pixel images and return a prediction in the range [0,9]. However, in the above architecture, you will see that those details are NOT hardcoded. Instead, we handle this by forcing the `LeNet` architecture to take placeholders for the images and annotations as inputs, and infer dimensions from those values. Specifically, given training data `(X_train,y_train)`, we can create a `LeNet` like this:

    x_shape = list(X_train.shape)
    x_shape[0] = None
    x = tf.placeholder(tf.float32, x_shape)

    y_shape = list(y_train.shape)
    y_shape[0] = None
    y = tf.placeholder(tf.float32, y_shape)

    lenet = LeNet(x, y)

In fact, the `LeNet` defined above is similar to the one defined in the `layers` package, with the distinction being that the `LeNet` in the `layers` package also utilizes dropout. Specifically, the `LeNet` class in the `layers` package implements dropout between the 3rd and 4th layers, and between the 4th and 5th layers:

    class LeNet:

        def __init__(self, x, y, depths=[10, 20, 200, 100]):

            self.dropout = tf.placeholder(tf.float32)
            ...

            x = FullLayer(name="layer_3", weight_shape=(    n    , depths[2])).output(x)
            x = tf.nn.dropout(x, keep_prob=self.dropout) # NEW!!!
            x = FullLayer(name="layer_4", weight_shape=(depths[2], depths[3])).output(x)
            x = tf.nn.dropout(x, keep_prob=self.dropout) # NEW!!!
            x = FullLayer(name="layer_5", weight_shape=(depths[3], y.get_shape().as_list()[1]), activation=lambda q: q).output(x)

            ...

To create and train a `LeNet` model on the training data `(X_train,y_train)`, you can use the following code snippet:

    x_shape = list(X_train.shape)
    x_shape[0] = None
    x = tf.placeholder(tf.float32, x_shape)

    y_shape = list(y_train.shape)
    y_shape[0] = None
    y = tf.placeholder(tf.float32, y_shape)

    lenet = LeNet(x, y)
    lenet.train(X_train, y_train)

Additionally, if you want to monitor the validation accuracy during the training process, you can pass that data to the `train` function:

    lenet.train(X_train, y_train, X_valid=X_valid, y_valid=y_valid)

Finally, some other parameters that you can pass to the `train` function are (defaults in parentheses):
1. `optimizer (tf.train.AdamOptimizer):` The optimizer used during training. Note that the learning rate may be dynamically set, so the optimizer must accept a `learning_rate` input. Specifically, the optimizer will be called like this

        trainer = optimizer(learning_rate=lr).minimize(self.cost)

    where the `trainer` will be called repeatedly in a loop to train the network.
1. `load_from (None):` The file location of the saved model variables. If `None`, then do not load from file.
1. `save_to (None):` The file location where to save the model variables. If `None`, then do not save.
1. `dropout (0.3):` The `keep_prob` in the `tf.nn.dropout` methods.
1. `X_valid (None):` The validation data. You can and should specify this so that you can monitor the validation accuracy while the model is training. If `None`, then do not monitor the validation accuracy.
1. `y_valid (None):` See `X_valid`.
1. `epochs (20):` The number of epochs to use when training the model.
1. `batch_size (1024):` The size of the batches used when training. Note that you may get an out of memory error if this value is too large.

## Traffic Signs

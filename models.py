from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self._train()

    def _train(self):
        return self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 sparse_inputs=True,
                                 logging=self.logging))

        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, layer_sizes, sparse_inputs=True, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = layer_sizes[0]
        self.sparse_inputs = sparse_inputs
        self.layer_sizes = layer_sizes
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'],
                                                  self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.y_pred = tf.argmax(self.predict(), 1)

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=self.layer_sizes[1],
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=self.sparse_inputs,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=self.layer_sizes[1],
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            sparse_inputs=self.sparse_inputs,
                                            logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)

class GCN_multipartite(GCN):
  def __init__(self, placeholders, layer_sizes, **kwargs):
    # actually the weights for computing the aggregate weights
    self.agg_layers = []
    self.feature_size = layer_sizes[0]

    super(GCN_multipartite, self).__init__(placeholders, layer_sizes, sparse_inputs=False, **kwargs)

  def aggregate(self, inputs):
    """ Aggregate features of neighbors using learned weights.

    Args:
      inputs: input features of dimension [batch size, max degree, feature length]

    Returns:
      Aggregated features of dimension [batch size, feature length].
    """
    
    dims = tf.shape(self.inputs)
    weights_inputs = tf.reshape(self.inputs, (dims[0] * dims[1], dims[2]))
    self.weights_activations = [weights_inputs]
    for layer in self.agg_layers:
      hidden = layer(self.weights_activations[-1])
      self.weights_activations.append(hidden)

    norm_agg_weights = tf.reshape(self.weights_activations[-1], (dims[0], dims[1], 1))
    # normalize (or softmax)
    norm_agg_weights = tf.nn.l2_normalize(norm_agg_weights, dim=1)

    features_trsp = tf.transpose(self.inputs, perm=[0, 2, 1])

    return tf.squeeze(tf.matmul(features_trsp, norm_agg_weights))

  def _build(self):

    agg_placeholders = {'dropout': self.placeholders['dropout']}
    self.agg_layers.append(Dense(input_dim=self.feature_size,
                                 output_dim=5,
                                 placeholders=agg_placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 logging=self.logging))

    self.agg_layers.append(Dense(input_dim=5,
                                 output_dim=1,
                                 placeholders=agg_placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 logging=self.logging))

    self.inputs = self.aggregate(self.inputs)

    super(GCN_multipartite, self)._build()


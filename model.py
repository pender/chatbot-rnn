import tensorflow as tf
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.framework import ops
from tensorflow.contrib import rnn

from tensorflow.python.util.nest import flatten

import numpy as np

class PartitionedMultiRNNCell(rnn_cell.RNNCell):
    """RNN cell composed sequentially of multiple simple cells."""

    # Diagramn of a PartitionedMultiRNNCell net with three layers and three partitions per layer.
    # Each brick shape is a partition, which comprises one RNNCell of size partition_size.
    # The two tilde (~) characters indicate wrapping (i.e. the two halves are a single partition).
    # Like laying bricks, each layer is offset by half a partition width so that influence spreads
    # horizontally through subsequent layers, while avoiding the quadratic resource scaling of fully
    # connected layers with respect to layer width.

    #        output
    #  //////// \\\\\\\\
    # -------------------
    # |     |     |     |
    # -------------------
    # ~  |     |     |  ~
    # -------------------
    # |     |     |     |
    # -------------------
    #  \\\\\\\\ ////////
    #        input


    def __init__(self, cell_fn, partition_size=128, partitions=1, layers=2):
        """Create a RNN cell composed sequentially of a number of RNNCells.
        Args:
            cell_fn: reference to RNNCell function to create each partition in each layer.
            partition_size: how many horizontal cells to include in each partition.
            partitions: how many horizontal partitions to include in each layer.
            layers: how many layers to include in the net.
        """
        super(PartitionedMultiRNNCell, self).__init__()

        self._cells = []
        for i in range(layers):
            self._cells.append([cell_fn(partition_size) for _ in range(partitions)])
        self._partitions = partitions

    @property
    def state_size(self):
        # Return a 2D tuple where each row is the partition's cell size repeated `partitions` times,
        # and there are `layers` rows of that.
        return tuple(((layer[0].state_size,) * len(layer)) for layer in self._cells)

    @property
    def output_size(self):
        # Return the output size of each partition in the last layer times the number of partitions per layer.
        return self._cells[-1][0].output_size * len(self._cells[-1])

    def zero_state(self, batch_size, dtype):
        # Return a 2D tuple of zero states matching the structure of state_size.
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return tuple(tuple(cell.zero_state(batch_size, dtype) for cell in layer) for layer in self._cells)

    def call(self, inputs, state):
        layer_input = inputs
        new_states = []
        for l, layer in enumerate(self._cells):
            # In between layers, offset the layer input by half of a partition width so that
            # activations can horizontally spread through subsequent layers.
            if l > 0:
                offset_width = layer[0].output_size // 2
                layer_input = tf.concat((layer_input[:, -offset_width:], layer_input[:, :-offset_width]),
                    axis=1, name='concat_offset_%d' % l)
            # Create a tuple of inputs by splitting the lower layer output into partitions.
            p_inputs = tf.split(layer_input, len(layer), axis=1, name='split_%d' % l)
            p_outputs = []
            p_states = []
            for p, p_inp in enumerate(p_inputs):
                with vs.variable_scope("cell_%d_%d" % (l, p)):
                    p_state = state[l][p]
                    cell = layer[p]
                    p_out, new_p_state = cell(p_inp, p_state)
                    p_outputs.append(p_out)
                    p_states.append(new_p_state)
            new_states.append(tuple(p_states))
            layer_input = tf.concat(p_outputs, axis=1, name='concat_%d' % l)
        new_states = tuple(new_states)
        return layer_input, new_states

def _rnn_state_placeholders(state):
    """Convert RNN state tensors to placeholders, reflecting the same nested tuple structure."""
    # Adapted from @carlthome's comment:
    # https://github.com/tensorflow/tensorflow/issues/2838#issuecomment-302019188
    if isinstance(state, tf.contrib.rnn.LSTMStateTuple):
        c, h = state
        c = tf.placeholder(c.dtype, c.shape, c.op.name)
        h = tf.placeholder(h.dtype, h.shape, h.op.name)
        return tf.contrib.rnn.LSTMStateTuple(c, h)
    elif isinstance(state, tf.Tensor):
        h = state
        h = tf.placeholder(h.dtype, h.shape, h.op.name)
        return h
    else:
        structure = [_rnn_state_placeholders(x) for x in state]
        return tuple(structure)

class Model():
    def __init__(self, args, infer=False): # infer is set to true during sampling.
        self.args = args
        if infer:
            # Worry about one character at a time during sampling; no batching or BPTT.
            args.batch_size = 1
            args.seq_length = 1

        # Set cell_fn to the type of network cell we're creating -- RNN, GRU, LSTM or NAS.
        if args.model == 'rnn':
            cell_fn = rnn_cell.BasicRNNCell
        elif args.model == 'gru':
            cell_fn = rnn_cell.GRUCell
        elif args.model == 'lstm':
            cell_fn = rnn_cell.BasicLSTMCell
        elif args.model == 'nas':
            cell_fn = rnn.NASCell
        else:
            raise Exception("model type not supported: {}".format(args.model))

        # Create variables to track training progress.
        self.lr = tf.Variable(args.learning_rate, name="learning_rate", trainable=False)
        self.global_epoch_fraction = tf.Variable(0.0, name="global_epoch_fraction", trainable=False)
        self.global_seconds_elapsed = tf.Variable(0.0, name="global_seconds_elapsed", trainable=False)

        # Call tensorflow library tensorflow-master/tensorflow/python/ops/rnn_cell
        # to create a layer of block_size cells of the specified basic type (RNN/GRU/LSTM).
        # Use the same rnn_cell library to create a stack of these cells
        # of num_layers layers. Pass in a python list of these cells. 
        # cell = rnn_cell.MultiRNNCell([cell_fn(args.block_size) for _ in range(args.num_layers)])
        # cell = MyMultiRNNCell([cell_fn(args.block_size) for _ in range(args.num_layers)])
        cell = PartitionedMultiRNNCell(cell_fn, partitions=args.num_blocks,
            partition_size=args.block_size, layers=args.num_layers)

        # Create a TF placeholder node of 32-bit ints (NOT floats!),
        # of shape batch_size x seq_length. This shape matches the batches
        # (listed in x_batches and y_batches) constructed in create_batches in utils.py.
        # input_data will receive input batches.
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

        self.zero_state = cell.zero_state(args.batch_size, tf.float32)

        self.initial_state = _rnn_state_placeholders(self.zero_state)
        self._flattened_initial_state = flatten(self.initial_state)

        layer_size = args.block_size * args.num_blocks

        # Scope our new variables to the scope identifier string "rnnlm".
        with tf.variable_scope('rnnlm'):
            # Create new variable softmax_w and softmax_b for output.
            # softmax_w is a weights matrix from the top layer of the model (of size layer_size)
            # to the vocabulary output (of size vocab_size).
            softmax_w = tf.get_variable("softmax_w", [layer_size, args.vocab_size])
            # softmax_b is a bias vector of the ouput characters (of size vocab_size).
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            # Create new variable named 'embedding' to connect the character input to the base layer
            # of the RNN. Its role is the conceptual inverse of softmax_w.
            # It contains the trainable weights from the one-hot input vector to the lowest layer of RNN.
            embedding = tf.get_variable("embedding", [args.vocab_size, layer_size])
            # Create an embedding tensor with tf.nn.embedding_lookup(embedding, self.input_data).
            # This tensor has dimensions batch_size x seq_length x layer_size.
            inputs = tf.nn.embedding_lookup(embedding, self.input_data)

        # TODO: Check arguments parallel_iterations (default uses more memory and less time) and
        # swap_memory (default uses more memory but "minimal (or no) performance penalty")
        outputs, self.final_state = tf.nn.dynamic_rnn(cell, inputs,
                initial_state=self.initial_state, scope='rnnlm')
        # outputs has shape [batch_size, max_time, cell.output_size] because time_major == false.
        # Do we need to transpose the first two dimensions? (Answer: no, this ruins everything.)
        # outputs = tf.transpose(outputs, perm=[1, 0, 2])
        output = tf.reshape(outputs, [-1, layer_size])
        # Obtain logits node by applying output weights and biases to the output tensor.
        # Logits is a tensor of shape [(batch_size * seq_length) x vocab_size].
        # Recall that outputs is a 2D tensor of shape [(batch_size * seq_length) x layer_size],
        # and softmax_w is a 2D tensor of shape [layer_size x vocab_size].
        # The matrix product is therefore a new 2D tensor of [(batch_size * seq_length) x vocab_size].
        # In other words, that multiplication converts a loooong list of layer_size vectors
        # to a loooong list of vocab_size vectors.
        # Then add softmax_b (a single vocab-sized vector) to every row of that list.
        # That gives you the logits!
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        if infer:
            # Convert logits to probabilities. Probs isn't used during training! That node is never calculated.
            # Like logits, probs is a tensor of shape [(batch_size * seq_length) x vocab_size].
            # During sampling, this means it is of shape [1 x vocab_size].
            self.probs = tf.nn.softmax(self.logits)
        else:
            # Create a targets placeholder of shape batch_size x seq_length.
            # Targets will be what output is compared against to calculate loss.
            self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
            # seq2seq.sequence_loss_by_example returns 1D float Tensor containing the log-perplexity
            # for each sequence. (Size is batch_size * seq_length.)
            # Targets are reshaped from a [batch_size x seq_length] tensor to a 1D tensor, of the following layout:
            #   target character (batch 0, seq 0)
            #   target character (batch 0, seq 1)
            #   ...
            #   target character (batch 0, seq seq_len-1)
            #   target character (batch 1, seq 0)
            #   ...
            # These targets are compared to the logits to generate loss.
            # Logits: instead of a list of character indices, it's a list of character index probability vectors.
            # seq2seq.sequence_loss_by_example will do the work of generating losses by comparing the one-hot vectors
            # implicitly represented by the target characters against the probability distrutions in logits.
            # It returns a 1D float tensor (a vector) where item i is the log-perplexity of
            # the comparison of the ith logit distribution to the ith one-hot target vector.

            loss = nn_ops.sparse_softmax_cross_entropy_with_logits(
                labels=tf.reshape(self.targets, [-1]), logits=self.logits)

            # Cost is the arithmetic mean of the values of the loss tensor.
            # It is a single-element floating point tensor. This is what the optimizer seeks to minimize.
            self.cost = tf.reduce_mean(loss)
            # Create a tensorboard summary of our cost.
            tf.summary.scalar("cost", self.cost)

            tvars = tf.trainable_variables() # tvars is a python list of all trainable TF Variable objects.
            # tf.gradients returns a list of tensors of length len(tvars) where each tensor is sum(dy/dx).
            grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                     args.grad_clip)
            optimizer = tf.train.AdamOptimizer(self.lr) # Use ADAM optimizer.
            # Zip creates a list of tuples, where each tuple is (variable tensor, gradient tensor).
            # Training op nudges the variables along the gradient, with the given learning rate, using the ADAM optimizer.
            # This is the op that a training session should be instructed to perform.
            self.train_op = optimizer.apply_gradients(zip(grads, tvars))
            #self.train_op = optimizer.minimize(self.cost)
            self.summary_op = tf.summary.merge_all()

    def add_state_to_feed_dict(self, feed_dict, state):
        for i, tensor in enumerate(flatten(state)):
            feed_dict[self._flattened_initial_state[i]] = tensor

    def save_variables_list(self):
        # Return a list of the trainable variables created within the rnnlm model.
        # This consists of the two projection softmax variables (softmax_w and softmax_b),
        # embedding, and all of the weights and biases in the MultiRNNCell model.
        # Save only the trainable variables and the placeholders needed to resume training;
        # discard the rest, including optimizer state.
        save_vars = set(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnnlm'))
        save_vars.update({self.lr, self.global_epoch_fraction, self.global_seconds_elapsed})
        return list(save_vars)

    def forward_model(self, sess, state, input_sample):
        '''Run a forward pass. Return the updated hidden state and the output probabilities.'''
        shaped_input = np.array([[input_sample]], np.float32)
        inputs = {self.input_data: shaped_input}
        self.add_state_to_feed_dict(inputs, state)
        [probs, state] = sess.run([self.probs, self.final_state], feed_dict=inputs)
        return probs[0], state

    def trainable_parameter_count(self):
        total_parameters = 0
        for variable in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rnnlm'):
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        return total_parameters

import tensorflow as tf
import numpy as np

class Model():
    def __init__(self, args, infer=False): # infer is set to true during sampling.
        self.args = args
        if infer:
            # Worry about one character at a time during sampling; no batching or BPTT.
            args.batch_size = 1
            args.seq_length = 1

        # Set cell_fn to the type of network cell we're creating -- RNN, GRU or LSTM.
        if args.model == 'rnn':
            # cell_fn = tf.nn.rnn_cell.BasicRNNCell
            cell = tf.nn.rnn_cell.BasicRNNCell(args.rnn_size)
        elif args.model == 'gru':
            # cell_fn = tf.nn.rnn_cell.GRUCell
            cell = tf.nn.rnn_cell.GRUCell(args.rnn_size)
        elif args.model == 'lstm':
            # cell_fn = tf.nn.rnn_cell.BasicLSTMCell
            cell = tf.nn.rnn_cell.BasicLSTMCell(args.rnn_size, state_is_tuple=True)
        elif args.model == 'nas':
            cell = tf.contrib.rnn.NASCell(args.rnn_size)
        else:
            raise Exception("model type not supported: {}".format(args.model))

        # Call tensorflow library tensorflow-master/tensorflow/python/ops/rnn_cell
        # to create a layer of rnn_size cells of the specified basic type (RNN/GRU/LSTM).
        # cell = cell_fn(args.rnn_size, state_is_tuple=True)

        # Use the same rnn_cell library to create a stack of these cells
        # of num_layers layers. Pass in a python list of these cells.
        # (The [cell] * arg.num_layers syntax literally duplicates cell multiple times in
        # a list. The syntax is such that [5, 6] * 3 would return [5, 6, 5, 6, 5, 6].)
        # self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([cell] * args.num_layers, state_is_tuple=True)
        self.cell = cell = tf.nn.rnn_cell.MultiRNNCell([cell for _ in range(args.num_layers)], state_is_tuple=True)

        # Create two TF placeholder nodes of 32-bit ints (NOT floats!),
        # each of shape batch_size x seq_length. This shape matches the batches
        # (listed in x_batches and y_batches) constructed in create_batches in utils.py.
        # input_data will receive input batches, and targets will be what it compares against
        # to calculate loss.
        self.input_data = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])
        self.targets = tf.placeholder(tf.int32, [args.batch_size, args.seq_length])

        # Using the zero_state function in the RNNCell master class in rnn_cell library,
        # create a tensor of zeros such that we can swap it in for the network state at any time
        # to zero out the network's state.
        # State dimensions are: cell_fn state size (2 for LSTM) x rnn_size x num_layers.
        # So an LSTM network with 100 cells per layer and 3 layers would have a state size of 600,
        # and initial_state would have a dimension of none x 600.
        self.initial_state = self.cell.zero_state(args.batch_size, tf.float32)

        # Scope our new variables to the scope identifier string "rnnlm".
        with tf.variable_scope('rnnlm'):
            # Create new variable softmax_w and softmax_b for output.
            # softmax_w is a weights matrix from the top layer of the model (of size rnn_size)
            # to the vocabulary output (of size vocab_size).
            softmax_w = tf.get_variable("softmax_w", [args.rnn_size, args.vocab_size])
            # softmax_b is a bias vector of the ouput characters (of size vocab_size).
            softmax_b = tf.get_variable("softmax_b", [args.vocab_size])
            # [TODO: Why specify CPU? Same as the TF translation tutorial, but don't know why.]
            with tf.device("/cpu:0"):
                # Create new variable named 'embedding' to connect the character input to the base layer
                # of the RNN. Its role is the conceptual inverse of softmax_w.
                # It contains the trainable weights from the one-hot input vector to the lowest layer of RNN.
                embedding = tf.get_variable("embedding", [args.vocab_size, args.rnn_size])
                # Create an embedding tensor with tf.nn.embedding_lookup(embedding, self.input_data).
                # This tensor has dimensions batch_size x seq_length x rnn_size.
                # tf.split splits that embedding lookup tensor into seq_length tensors (along dimension 1).
                # Thus inputs is a list of seq_length different tensors,
                # each of dimension batch_size x 1 x rnn_size.
                inputs = tf.split(axis=1, num_or_size_splits=args.seq_length, value=tf.nn.embedding_lookup(embedding, self.input_data))
                # Iterate through these resulting tensors and eliminate that degenerate second dimension of 1,
                # i.e. squeeze each from batch_size x 1 x rnn_size down to batch_size x rnn_size.
                # Thus we now have a list of seq_length tensors, each with dimension batch_size x rnn_size.
                inputs = [tf.squeeze(input_, [1]) for input_ in inputs]

        # THIS LOOP FUNCTION IS NEVER ACTUALLY USED.
        # IT IS EXPLICITLY NOT USED DURING TRAINING.
        # DURING INFERENCE, SEQ_LENGTH == 1, SO SEQ2SEQ.RNN_DECODER() ONLY USES THE LOOP ARGUMENT
        # ON SEQUENCE LENGTH ITEMS SUBSEQUENT TO THE FIRST.
        # This looping function is used as part of seq2seq.rnn_decoder only during sampling -- not training.
        # prev is a 2D Tensor of shape [batch_size x cell.output_size].
        # returns a 2D Tensor of shape [batch_size x cell.input_size].
        def loop(prev, _):
            # prev is initially the top cell state.
            # Convert the top cell state into character logits.
            prev = tf.matmul(prev, softmax_w) + softmax_b
            # Pull the character with the greatest logit (no sampling, just argmaxing).
            # WHY IS THIS ARGMAXING WHEN ACTUAL SAMPLING IS DONE PROBABILISTICALLY?
            # DOESN'T THIS CAUSE OUTPUTS NOT TO MATCH INPUTS DURING SEQUENCE GENERATION?
            prev_symbol = tf.stop_gradient(tf.argmax(prev, 1))
            # Re-embed that symbol as the next step's input, and return that.
            return tf.nn.embedding_lookup(embedding, prev_symbol)

        # Set up a seq2seq decoder from the seq2seq.py library.
        # This constructs the outputs and states nodes of the network.
        # Outputs is a list (of len seq_length, same as inputs) of tensors of shape [batch_size x rnn_size].
        # These are the raw output values of the top layer of the network at each time step.
        # They have NOT been fed through the decoder projection; they are still in network space,
        # not character space.
        # State is a tensor of shape [batch_size x cell.state_size].
        # This is also the step where all of the trainable parameters for the LSTM (weights and biases) are defined.
        outputs, self.final_state = tf.contrib.legacy_seq2seq.rnn_decoder(inputs,
                self.initial_state, cell, loop_function=loop if infer else None, scope='rnnlm')
        # tf.concat concatenates the output tensors along the rnn_size dimension,
        # to make a single tensor of shape [batch_size x (seq_length * rnn_size)].
        # This gives the following 2D outputs matrix:
        #   [(rnn output: batch 0, seq 0) (rnn output: batch 0, seq 1) ... (rnn output: batch 0, seq seq_len-1)]
        #   [(rnn output: batch 1, seq 0) (rnn output: batch 1, seq 1) ... (rnn output: batch 1, seq seq_len-1)]
        #   ...
        #   [(rnn output: batch batch_size-1, seq 0) (rnn output: batch batch_size-1, seq 1) ... (rnn output: batch batch_size-1, seq seq_len-1)]
        # tf.reshape then reshapes it to a tensor of shape [(batch_size * seq_length) x rnn_size].
        # Output will now be the following matrix:
        #   [rnn output: batch 0, seq 0]
        #   [rnn output: batch 0, seq 1]
        #   ...
        #   [rnn output: batch 0, seq seq_len-1]
        #   [rnn output: batch 1, seq 0]
        #   [rnn output: batch 1, seq 1]
        #   ...
        #   [rnn output: batch 1, seq seq_len-1]
        #   ...
        #   ...
        #   [rnn output: batch batch_size-1, seq seq_len-1]
        # Note the following comment in rnn_cell.py:
        #   Note: in many cases it may be more efficient to not use this wrapper,
        #   but instead concatenate the whole sequence of your outputs in time,
        #   do the projection on this batch-concatenated sequence, then split it
        #   if needed or directly feed into a softmax.
        output = tf.reshape(tf.concat(axis=1, values=outputs), [-1, args.rnn_size])
        # Obtain logits node by applying output weights and biases to the output tensor.
        # Logits is a tensor of shape [(batch_size * seq_length) x vocab_size].
        # Recall that outputs is a 2D tensor of shape [(batch_size * seq_length) x rnn_size],
        # and softmax_w is a 2D tensor of shape [rnn_size x vocab_size].
        # The matrix product is therefore a new 2D tensor of [(batch_size * seq_length) x vocab_size].
        # In other words, that multiplication converts a loooong list of rnn_size vectors
        # to a loooong list of vocab_size vectors.
        # Then add softmax_b (a single vocab-sized vector) to every row of that list.
        # That gives you the logits!
        self.logits = tf.matmul(output, softmax_w) + softmax_b
        # Convert logits to probabilities. Probs isn't used during training! That node is never calculated.
        # Like logits, probs is a tensor of shape [(batch_size * seq_length) x vocab_size].
        # During sampling, this means it is of shape [1 x vocab_size].
        self.probs = tf.nn.softmax(self.logits)
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
        loss = tf.contrib.legacy_seq2seq.sequence_loss_by_example([self.logits], # logits: 1-item list of 2D Tensors of shape [batch_size x vocab_size]
                [tf.reshape(self.targets, [-1])], # targets: 1-item list of 1D batch-sized int32 Tensors of the same length as logits
                [tf.ones([args.batch_size * args.seq_length])], # weights: 1-item list of 1D batch-sized float-Tensors of the same length as logits
                args.vocab_size) # num_decoder_symbols: integer, number of decoder symbols (output classes)
        # Cost is the arithmetic mean of the values of the loss tensor
        # (the sum divided by the total number of elements).
        # It is a single-element floating point tensor. This is what the optimizer seeks to minimize.
        self.cost = tf.reduce_sum(loss) / args.batch_size / args.seq_length
        # Create a summary for our cost.
        # tf.scalar_summary("cost", self.cost)
        tf.summary.scalar("cost", self.cost)
        # Create a node to track the learning rate as it decays through the epochs.
        self.lr = tf.Variable(args.learning_rate, trainable=False)
        self.global_epoch_fraction = tf.Variable(0.0, trainable=False)
        self.global_seconds_elapsed = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables() # tvars is a python list of all trainable TF Variable objects.

        # tf.gradients returns a list of tensors of length len(tvars) where each tensor is sum(dy/dx).
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars),
                args.grad_clip)
        optimizer = tf.train.AdamOptimizer(self.lr) # Use ADAM optimizer with the current learning rate.
        # Zip creates a list of tuples, where each tuple is (variable tensor, gradient tensor).
        # Training op nudges the variables along the gradient, with the given learning rate, using the ADAM optimizer.
        # This is the op that a training session should be instructed to perform.
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        # self.summary_op = tf.merge_all_summaries()
        self.summary_op = tf.summary.merge_all()

    def save_variables_list(self):
        # Return a list of the trainable variables created within the rnnlm model.
        # This consists of the two projection softmax variables (softmax_w and softmax_b),
        # embedding, and all of the weights and biases in the MultiRNNCell model.
        # Save only the trainable variables and the placeholders needed to resume training;
        # discard the rest, including optimizer state.
        save_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='rnnlm')
        save_vars.append(self.lr)
        save_vars.append(self.global_epoch_fraction)
        save_vars.append(self.global_seconds_elapsed)
        return save_vars

    def forward_model(self, sess, state, input_sample):
        '''Run a forward pass. Return the updated hidden state and the output probabilities.'''
        shaped_input = np.array([[input_sample]], np.float32)
        inputs = {self.input_data: shaped_input,
                    self.initial_state: state}
        [probs, state] = sess.run([self.probs, self.final_state], feed_dict=inputs)
        return probs[0], state

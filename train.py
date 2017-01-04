import numpy as np
import tensorflow as tf

import argparse
import time
import os
import cPickle

from utils import TextLoader
from model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/scotus',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='models/new_save',
                       help='directory for checkpointed models (load from here if one is already present)')
    parser.add_argument('--rnn_size', type=int, default=1500,
                       help='size of RNN hidden state')
    parser.add_argument('--num_layers', type=int, default=4,
                       help='number of layers in the RNN')
    parser.add_argument('--model', type=str, default='gru',
                       help='rnn, gru, or lstm')
    parser.add_argument('--batch_size', type=int, default=40,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=50,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=1000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=6e-5,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.95,
                       help='how much to decay the learning rate')
    parser.add_argument('--decay_steps', type=int, default=100000,
                       help='how often to decay the learning rate')
    args = parser.parse_args()
    train(args)

def train(args):
    # Create the data_loader object, which loads up all of our batches, vocab dictionary, etc.
    # from utils.py (and creates them if they don't already exist).
    # These files go in the data directory.
    data_loader = TextLoader(args.data_dir, args.batch_size, args.seq_length)
    args.vocab_size = data_loader.vocab_size

    load_model = False
    if not os.path.exists(args.save_dir):
        print("Creating directory %s" % args.save_dir)
        os.mkdir(args.save_dir)
    elif (os.path.exists(os.path.join(args.save_dir, 'config.pkl'))):
        # Trained model already exists
        ckpt = tf.train.get_checkpoint_state(args.save_dir)
        if ckpt and ckpt.model_checkpoint_path:
            with open(os.path.join(args.save_dir, 'config.pkl')) as f:
                saved_args = cPickle.load(f)
                args.rnn_size = saved_args.rnn_size
                args.num_layers = saved_args.num_layers
                args.model = saved_args.model
                print("Found a previous checkpoint. Overwriting model description arguments to:")
                print(" model: {}, rnn_size: {}, num_layers: {}".format(
                    saved_args.model, saved_args.rnn_size, saved_args.num_layers))
                load_model = True

    # Save all arguments to config.pkl in the save directory -- NOT the data directory.
    with open(os.path.join(args.save_dir, 'config.pkl'), 'w') as f:
        cPickle.dump(args, f)
    # Save a tuple of the characters list and the vocab dictionary to chars_vocab.pkl in
    # the save directory -- NOT the data directory.
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'w') as f:
        cPickle.dump((data_loader.chars, data_loader.vocab), f)

    # Create the model!
    print("Building the model")
    model = Model(args)

    config = tf.ConfigProto(log_device_placement=False)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(model.save_variables_list())
        if (load_model):
            print("Loading saved parameters")
            saver.restore(sess, ckpt.model_checkpoint_path)
        global_epoch_fraction = sess.run(model.global_epoch_fraction)
        global_seconds_elapsed = sess.run(model.global_seconds_elapsed)
        if load_model: print("Resuming from global epoch fraction {:.3f},"
                " total trained time: {}, learning rate: {}".format(
                global_epoch_fraction, global_seconds_elapsed, sess.run(model.lr)))
        data_loader.cue_batch_pointer_to_epoch_fraction(global_epoch_fraction)
        initial_batch_step = int((global_epoch_fraction
                - int(global_epoch_fraction)) * data_loader.total_batch_count)
        epoch_range = (int(global_epoch_fraction),
                args.num_epochs + int(global_epoch_fraction))
        writer = tf.train.SummaryWriter(args.save_dir, graph=tf.get_default_graph())
        outputs = [model.cost, model.final_state, model.train_op, model.summary_op]
        is_lstm = args.model == 'lstm'
        global_step = epoch_range[0] * data_loader.total_batch_count + initial_batch_step
        try:
            for e in xrange(*epoch_range):
                # e iterates through the training epochs.
                # Reset the model state, so it does not carry over from the end of the previous epoch.
                state = sess.run(model.initial_state)
                batch_range = (initial_batch_step, data_loader.total_batch_count)
                initial_batch_step = 0
                for b in xrange(*batch_range):
                    global_step += 1
                    if global_step % args.decay_steps == 0:
                        # Set the model.lr element of the model to track
                        # the appropriately decayed learning rate.
                        current_learning_rate = sess.run(model.lr)
                        current_learning_rate *= args.decay_rate
                        sess.run(tf.assign(model.lr, current_learning_rate))
                        print("Decayed learning rate to {}".format(current_learning_rate))
                    start = time.time()
                    # Pull the next batch inputs (x) and targets (y) from the data loader.
                    x, y = data_loader.next_batch()

                    # feed is a dictionary of variable references and respective values for initialization.
                    # Initialize the model's input data and target data from the batch,
                    # and initialize the model state to the final state from the previous batch, so that
                    # model state is accumulated and carried over between batches.
                    feed = {model.input_data: x, model.targets: y}
                    if is_lstm:
                        for i, (c, h) in enumerate(model.initial_state):
                            feed[c] = state[i].c
                            feed[h] = state[i].h
                    else:
                        for i, c in enumerate(model.initial_state):
                            feed[c] = state[i]
                    # Run the session! Specifically, tell TensorFlow to compute the graph to calculate
                    # the values of cost, final state, and the training op.
                    # Cost is used to monitor progress.
                    # Final state is used to carry over the state into the next batch.
                    # Training op is not used, but we want it to be calculated, since that calculation
                    # is what updates parameter states (i.e. that is where the training happens).
                    train_loss, state, _, summary = sess.run(outputs, feed)
                    elapsed = time.time() - start
                    global_seconds_elapsed += elapsed
                    writer.add_summary(summary, e * batch_range[1] + b + 1)
                    print "{}/{} (epoch {}/{}), loss = {:.3f}, time/batch = {:.3f}s" \
                        .format(b, batch_range[1], e, epoch_range[1], train_loss, elapsed)
                    # Every save_every batches, save the model to disk.
                    # By default, only the five most recent checkpoint files are kept.
                    if (e * batch_range[1] + b + 1) % args.save_every == 0 \
                            or (e == epoch_range[1] - 1 and b == batch_range[1] - 1):
                        save_model(sess, saver, model, args.save_dir, global_step,
                                data_loader.total_batch_count, global_seconds_elapsed)
        except KeyboardInterrupt:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
            print()
        finally:
            writer.flush()
            global_step = e * data_loader.total_batch_count + b
            save_model(sess, saver, model, args.save_dir, global_step,
                    data_loader.total_batch_count, global_seconds_elapsed)

def save_model(sess, saver, model, save_dir, global_step, steps_per_epoch, global_seconds_elapsed):
    global_epoch_fraction = float(global_step) / float(steps_per_epoch)
    checkpoint_path = os.path.join(save_dir, 'model.ckpt')
    print "Saving model to {} (epoch fraction {:.3f})".format(checkpoint_path, global_epoch_fraction)
    sess.run(tf.assign(model.global_epoch_fraction, global_epoch_fraction))
    sess.run(tf.assign(model.global_seconds_elapsed, global_seconds_elapsed))
    saver.save(sess, checkpoint_path, global_step = global_step)
    print "Model saved."

if __name__ == '__main__':
    main()

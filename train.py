import numpy as np
import tensorflow as tf

import argparse
import time, datetime
import os
import pickle
import sys

from utils import TextLoader
from model import Model

def main():
    assert sys.version_info >= (3, 3), \
    "Must be run in Python 3.3 or later. You are running {}".format(sys.version)

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/scotus',
                       help='data directory containing input.txt')
    parser.add_argument('--save_dir', type=str, default='models/new_save',
                       help='directory for checkpointed models (load from here if one is already present)')
    parser.add_argument('--block_size', type=int, default=2048,
                       help='number of cells per block')
    parser.add_argument('--num_blocks', type=int, default=3,
                       help='number of blocks per layer')
    parser.add_argument('--num_layers', type=int, default=3,
                       help='number of layers')
    parser.add_argument('--model', type=str, default='gru',
                       help='rnn, gru, lstm or nas')
    parser.add_argument('--batch_size', type=int, default=40,
                       help='minibatch size')
    parser.add_argument('--seq_length', type=int, default=40,
                       help='RNN sequence length')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='number of epochs')
    parser.add_argument('--save_every', type=int, default=5000,
                       help='save frequency')
    parser.add_argument('--grad_clip', type=float, default=5.,
                       help='clip gradients at this value')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                       help='learning rate')
    parser.add_argument('--decay_rate', type=float, default=0.975,
                       help='how much to decay the learning rate')
    parser.add_argument('--decay_steps', type=int, default=100000,
                       help='how often to decay the learning rate')
    parser.add_argument('--set_learning_rate', type=float, default=-1,
                       help='reset learning rate to this value (if greater than zero)')
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
            with open(os.path.join(args.save_dir, 'config.pkl'), 'rb') as f:
                saved_args = pickle.load(f)
                args.block_size = saved_args.block_size
                args.num_blocks = saved_args.num_blocks
                args.num_layers = saved_args.num_layers
                args.model = saved_args.model
                print("Found a previous checkpoint. Overwriting model description arguments to:")
                print(" model: {}, block_size: {}, num_blocks: {}, num_layers: {}".format(
                    saved_args.model, saved_args.block_size, saved_args.num_blocks, saved_args.num_layers))
                load_model = True

    # Save all arguments to config.pkl in the save directory -- NOT the data directory.
    with open(os.path.join(args.save_dir, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)
    # Save a tuple of the characters list and the vocab dictionary to chars_vocab.pkl in
    # the save directory -- NOT the data directory.
    with open(os.path.join(args.save_dir, 'chars_vocab.pkl'), 'wb') as f:
        pickle.dump((data_loader.chars, data_loader.vocab), f)

    # Create the model!
    print("Building the model")
    model = Model(args)
    print("Total trainable parameters: {:,d}".format(model.trainable_parameter_count()))
    
    # Make tensorflow less verbose; filter out info (1+) and warnings (2+) but not errors (3).
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    config = tf.ConfigProto(log_device_placement=False)
    #config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(model.save_variables_list(), max_to_keep=3)
        if (load_model):
            print("Loading saved parameters")
            saver.restore(sess, ckpt.model_checkpoint_path)
        global_epoch_fraction = sess.run(model.global_epoch_fraction)
        global_seconds_elapsed = sess.run(model.global_seconds_elapsed)
        if load_model: print("Resuming from global epoch fraction {:.3f},"
                " total trained time: {}, learning rate: {}".format(
                global_epoch_fraction,
                datetime.timedelta(seconds=float(global_seconds_elapsed)),
                sess.run(model.lr)))
        if (args.set_learning_rate > 0):
            sess.run(tf.assign(model.lr, args.set_learning_rate))
            print("Reset learning rate to {}".format(args.set_learning_rate))
        data_loader.cue_batch_pointer_to_epoch_fraction(global_epoch_fraction)
        initial_batch_step = int((global_epoch_fraction
                - int(global_epoch_fraction)) * data_loader.total_batch_count)
        epoch_range = (int(global_epoch_fraction),
                args.num_epochs + int(global_epoch_fraction))
        writer = tf.summary.FileWriter(args.save_dir, graph=tf.get_default_graph())
        outputs = [model.cost, model.final_state, model.train_op, model.summary_op]
        global_step = epoch_range[0] * data_loader.total_batch_count + initial_batch_step
        avg_loss = 0
        avg_steps = 0
        try:
            for e in range(*epoch_range):
                # e iterates through the training epochs.
                # Reset the model state, so it does not carry over from the end of the previous epoch.
                state = sess.run(model.zero_state)
                batch_range = (initial_batch_step, data_loader.total_batch_count)
                initial_batch_step = 0
                for b in range(*batch_range):
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
                    model.add_state_to_feed_dict(feed, state)
                    
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
                    if avg_steps < 100: avg_steps += 1
                    avg_loss = 1 / avg_steps * train_loss + (1 - 1 / avg_steps) * avg_loss
                    print("{:,d} / {:,d} (epoch {:.3f} / {}), loss {:.3f} (avg {:.3f}), {:.3f}s" \
                        .format(b, batch_range[1], e + b / batch_range[1], epoch_range[1],
                            train_loss, avg_loss, elapsed))
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
    print("Saving model to {} (epoch fraction {:.3f})...".format(checkpoint_path, global_epoch_fraction),
        end='', flush=True)
    sess.run(tf.assign(model.global_epoch_fraction, global_epoch_fraction))
    sess.run(tf.assign(model.global_seconds_elapsed, global_seconds_elapsed))
    saver.save(sess, checkpoint_path, global_step = global_step)
    print("\rSaved model to {} (epoch fraction {:.3f}).   ".format(checkpoint_path, global_epoch_fraction))

if __name__ == '__main__':
    main()

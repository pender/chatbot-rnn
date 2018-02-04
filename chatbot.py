from __future__ import print_function

import numpy as np
import tensorflow as tf

import argparse
import os
import pickle
import copy
import sys
import html

from utils import TextLoader
from model import Model

def main():
    assert sys.version_info >= (3, 3), \
    "Must be run in Python 3.3 or later. You are running {}".format(sys.version)
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='models/reddit',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=500,
                       help='number of characters to sample')
    parser.add_argument('--prime', type=str, default=' ',
                       help='prime text')
    parser.add_argument('--beam_width', type=int, default=2,
                       help='Width of the beam for beam search, default 2')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='sampling temperature'
                       '(lower is more conservative, default is 1.0, which is neutral)')
    parser.add_argument('--topn', type=int, default=-1,
                        help='at each step, choose from only this many most likely characters;'
                        'set to <0 to disable top-n filtering.')
    parser.add_argument('--relevance', type=float, default=-1.,
                       help='amount of "relevance masking/MMI (disabled by default):"'
                       'higher is more pressure, 0.4 is probably as high as it can go without'
                       'noticeably degrading coherence;'
                       'set to <0 to disable relevance masking')
    args = parser.parse_args()
    sample_main(args)

def get_paths(input_path):
    if os.path.isfile(input_path):
        # Passed a model rather than a checkpoint directory
        model_path = input_path
        save_dir = os.path.dirname(model_path)
    elif os.path.exists(input_path):
        # Passed a checkpoint directory
        save_dir = input_path
        checkpoint = tf.train.get_checkpoint_state(save_dir)
        if checkpoint:
            model_path = checkpoint.model_checkpoint_path
        else:
            raise ValueError('Checkpoint not found in {}.'.format(save_dir))
    else:
        raise ValueError('save_dir is not a valid path.')
    return model_path, os.path.join(save_dir, 'config.pkl'), os.path.join(save_dir, 'chars_vocab.pkl')

def sample_main(args):
    model_path, config_path, vocab_path = get_paths(args.save_dir)
    # Arguments passed to sample.py direct us to a saved model.
    # Load the separate arguments by which that model was previously trained.
    # That's saved_args. Use those to load the model.
    with open(config_path, 'rb') as f:
        saved_args = pickle.load(f)
    # Separately load chars and vocab from the save directory.
    with open(vocab_path, 'rb') as f:
        chars, vocab = pickle.load(f)
    # Create the model from the saved arguments, in inference mode.
    print("Creating model...")
    saved_args.batch_size = args.beam_width
    net = Model(saved_args, True)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Make tensorflow less verbose; filter out info (1+) and warnings (2+) but not errors (3).
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    with tf.Session(config=config) as sess:
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(net.save_variables_list())
        # Restore the saved variables, replacing the initialized values.
        print("Restoring weights...")
        saver.restore(sess, model_path)
        chatbot(net, sess, chars, vocab, args.n, args.beam_width,
                args.relevance, args.temperature, args.topn)

def initial_state(net, sess):
    # Return freshly initialized model states.
    return sess.run(net.zero_state)

def forward_text(net, sess, states, relevance, vocab, prime_text=None):
    if prime_text is not None:
        for char in prime_text:
            if relevance > 0.:
                # Automatically forward the primary net.
                _, states[0] = net.forward_model(sess, states[0], vocab[char])
                # If the token is newline, reset the mask net state; else, forward it.
                if vocab[char] == '\n':
                    states[1] = initial_state(net, sess)
                else:
                    _, states[1] = net.forward_model(sess, states[1], vocab[char])
            else:
                _, states = net.forward_model(sess, states, vocab[char])
    return states

def sanitize_text(vocab, text): # Strip out characters that are not part of the net's vocab.
    return ''.join(i for i in text if i in vocab)

def initial_state_with_relevance_masking(net, sess, relevance):
    if relevance <= 0.: return initial_state(net, sess)
    else: return [initial_state(net, sess), initial_state(net, sess)]

def possibly_escaped_char(raw_chars):
    if raw_chars[-1] == ';':
        for i, c in enumerate(reversed(raw_chars[:-1])):
            if c == ';' or i > 8:
                return raw_chars[-1]
            elif c == '&':
                escape_seq = "".join(raw_chars[-(i + 2):])
                new_seq = html.unescape(escape_seq)
                backspace_seq = "".join(['\b'] * (len(escape_seq)-1))
                diff_length = len(escape_seq) - len(new_seq) - 1
                return backspace_seq + new_seq + "".join([' '] * diff_length) + "".join(['\b'] * diff_length)
    return raw_chars[-1]

def chatbot(net, sess, chars, vocab, max_length, beam_width, relevance, temperature, topn):
    states = initial_state_with_relevance_masking(net, sess, relevance)
    while True:
        user_input = input('\n> ')
        user_command_entered, reset, states, relevance, temperature, topn, beam_width = process_user_command(
            user_input, states, relevance, temperature, topn, beam_width)
        if reset: states = initial_state_with_relevance_masking(net, sess, relevance)
        if not user_command_entered:
            states = forward_text(net, sess, states, relevance, vocab, sanitize_text(vocab, "> " + user_input + "\n>"))
            computer_response_generator = beam_search_generator(sess=sess, net=net,
                initial_state=copy.deepcopy(states), initial_sample=vocab[' '],
                early_term_token=vocab['\n'], beam_width=beam_width, forward_model_fn=forward_with_mask,
                forward_args={'relevance':relevance, 'mask_reset_token':vocab['\n'], 'forbidden_token':vocab['>'],
                                'temperature':temperature, 'topn':topn})
            out_chars = []
            for i, char_token in enumerate(computer_response_generator):
                out_chars.append(chars[char_token])
                print(possibly_escaped_char(out_chars), end='', flush=True)
                states = forward_text(net, sess, states, relevance, vocab, chars[char_token])
                if i >= max_length: break
            states = forward_text(net, sess, states, relevance, vocab, sanitize_text(vocab, "\n> "))

def process_user_command(user_input, states, relevance, temperature, topn, beam_width):
    user_command_entered = False
    reset = False
    try:
        if user_input.startswith('--temperature '):
            user_command_entered = True
            temperature = max(0.001, float(user_input[len('--temperature '):]))
            print("[Temperature set to {}]".format(temperature))
        elif user_input.startswith('--relevance '):
            user_command_entered = True
            new_relevance = float(user_input[len('--relevance '):])
            if relevance <= 0. and new_relevance > 0.:
                states = [states, copy.deepcopy(states)]
            elif relevance > 0. and new_relevance <= 0.:
                states = states[0]
            relevance = new_relevance
            print("[Relevance disabled]" if relevance <= 0. else "[Relevance set to {}]".format(relevance))
        elif user_input.startswith('--topn '):
            user_command_entered = True
            topn = int(user_input[len('--topn '):])
            print("[Top-n filtering disabled]" if topn <= 0 else "[Top-n filtering set to {}]".format(topn))
        elif user_input.startswith('--beam_width '):
            user_command_entered = True
            beam_width = max(1, int(user_input[len('--beam_width '):]))
            print("[Beam width set to {}]".format(beam_width))
        elif user_input.startswith('--reset'):
            user_command_entered = True
            reset = True
            print("[Model state reset]")
    except ValueError:
        print("[Value error with provided argument.]")
    return user_command_entered, reset, states, relevance, temperature, topn, beam_width

def consensus_length(beam_outputs, early_term_token):
    for l in range(len(beam_outputs[0])):
        if l > 0 and beam_outputs[0][l-1] == early_term_token:
            return l-1, True
        for b in beam_outputs[1:]:
            if beam_outputs[0][l] != b[l]: return l, False
    return l, False

def scale_prediction(prediction, temperature):
    if (temperature == 1.0): return prediction # Temperature 1.0 makes no change
    np.seterr(divide='ignore')
    scaled_prediction = np.log(prediction) / temperature
    scaled_prediction = scaled_prediction - np.logaddexp.reduce(scaled_prediction)
    scaled_prediction = np.exp(scaled_prediction)
    np.seterr(divide='warn')
    return scaled_prediction

def forward_with_mask(sess, net, states, input_sample, forward_args):
    # forward_args is a dictionary containing arguments for generating probabilities.
    relevance = forward_args['relevance']
    mask_reset_token = forward_args['mask_reset_token']
    forbidden_token = forward_args['forbidden_token']
    temperature = forward_args['temperature']
    topn = forward_args['topn']

    if relevance <= 0.:
        # No relevance masking.
        prob, states = net.forward_model(sess, states, input_sample)
    else:
        # states should be a 2-length list: [primary net state, mask net state].
        if input_sample == mask_reset_token:
            # Reset the mask probs when reaching mask_reset_token (newline).
            states[1] = initial_state(net, sess)
        primary_prob, states[0] = net.forward_model(sess, states[0], input_sample)
        primary_prob /= sum(primary_prob)
        mask_prob, states[1] = net.forward_model(sess, states[1], input_sample)
        mask_prob /= sum(mask_prob)
        prob = np.exp(np.log(primary_prob) - relevance * np.log(mask_prob))
    # Mask out the forbidden token (">") to prevent the bot from deciding the chat is over)
    prob[forbidden_token] = 0
    # Normalize probabilities so they sum to 1.
    prob = prob / sum(prob)
    # Apply temperature.
    prob = scale_prediction(prob, temperature)
    # Apply top-n filtering if enabled
    if topn > 0:
        prob[np.argsort(prob)[:-topn]] = 0
        prob = prob / sum(prob)
    return prob, states

def beam_search_generator(sess, net, initial_state, initial_sample,
    early_term_token, beam_width, forward_model_fn, forward_args):
    '''Run beam search! Yield consensus tokens sequentially, as a generator;
    return when reaching early_term_token (newline).

    Args:
        sess: tensorflow session reference
        net: tensorflow net graph (must be compatible with the forward_net function)
        initial_state: initial hidden state of the net
        initial_sample: single token (excluding any seed/priming material)
            to start the generation
        early_term_token: stop when the beam reaches consensus on this token
            (but do not return this token).
        beam_width: how many beams to track
        forward_model_fn: function to forward the model, must be of the form:
            probability_output, beam_state =
                    forward_model_fn(sess, net, beam_state, beam_sample, forward_args)
            (Note: probability_output has to be a valid probability distribution!)
        tot_steps: how many tokens to generate before stopping,
            unless already stopped via early_term_token.
    Returns: a generator to yield a sequence of beam-sampled tokens.'''
    # Store state, outputs and probabilities for up to args.beam_width beams.
    # Initialize with just the one starting entry; it will branch to fill the beam
    # in the first step.
    beam_states = [initial_state] # Stores the best activation states
    beam_outputs = [[initial_sample]] # Stores the best generated output sequences so far.
    beam_probs = [1.] # Stores the cumulative normalized probabilities of the beams so far.

    while True:
        # Keep a running list of the best beam branches for next step.
        # Don't actually copy any big data structures yet, just keep references
        # to existing beam state entries, and then clone them as necessary
        # at the end of the generation step.
        new_beam_indices = []
        new_beam_probs = []
        new_beam_samples = []

        # Iterate through the beam entries.
        for beam_index, beam_state in enumerate(beam_states):
            beam_prob = beam_probs[beam_index]
            beam_sample = beam_outputs[beam_index][-1]

            # Forward the model.
            prediction, beam_states[beam_index] = forward_model_fn(
                    sess, net, beam_state, beam_sample, forward_args)

            # Sample best_tokens from the probability distribution.
            # Sample from the scaled probability distribution beam_width choices
            # (but not more than the number of positive probabilities in scaled_prediction).
            count = min(beam_width, sum(1 if p > 0. else 0 for p in prediction))
            best_tokens = np.random.choice(len(prediction), size=count,
                                            replace=False, p=prediction)
            for token in best_tokens:
                prob = prediction[token] * beam_prob
                if len(new_beam_indices) < beam_width:
                    # If we don't have enough new_beam_indices, we automatically qualify.
                    new_beam_indices.append(beam_index)
                    new_beam_probs.append(prob)
                    new_beam_samples.append(token)
                else:
                    # Sample a low-probability beam to possibly replace.
                    np_new_beam_probs = np.array(new_beam_probs)
                    inverse_probs = -np_new_beam_probs + max(np_new_beam_probs) + min(np_new_beam_probs)
                    inverse_probs = inverse_probs / sum(inverse_probs)
                    sampled_beam_index = np.random.choice(beam_width, p=inverse_probs)
                    if new_beam_probs[sampled_beam_index] <= prob:
                        # Replace it.
                        new_beam_indices[sampled_beam_index] = beam_index
                        new_beam_probs[sampled_beam_index] = prob
                        new_beam_samples[sampled_beam_index] = token
        # Replace the old states with the new states, first by referencing and then by copying.
        already_referenced = [False] * beam_width
        new_beam_states = []
        new_beam_outputs = []
        for i, new_index in enumerate(new_beam_indices):
            if already_referenced[new_index]:
                new_beam = copy.deepcopy(beam_states[new_index])
            else:
                new_beam = beam_states[new_index]
                already_referenced[new_index] = True
            new_beam_states.append(new_beam)
            new_beam_outputs.append(beam_outputs[new_index] + [new_beam_samples[i]])
        # Normalize the beam probabilities so they don't drop to zero
        beam_probs = new_beam_probs / sum(new_beam_probs)
        beam_states = new_beam_states
        beam_outputs = new_beam_outputs
        # Prune the agreed portions of the outputs
        # and yield the tokens on which the beam has reached consensus.
        l, early_term = consensus_length(beam_outputs, early_term_token)
        if l > 0:
            for token in beam_outputs[0][:l]: yield token
            beam_outputs = [output[l:] for output in beam_outputs]
        if early_term: return

if __name__ == '__main__':
    main()

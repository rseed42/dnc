import os
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.layers.python.layers.optimizers import optimize_loss, OPTIMIZER_SUMMARIES

from ops_tf import *
from ops_np import *

class DNC(object):
    def __init__(self, dataset, params):
        self.X_train = dataset.training_data
        self.y_train = dataset.training_labels
        self.X_test = dataset.testing_data
        self.y_test = dataset.testing_labels

        self.N = params.N
        self.W = params.W
        self.R = params.R
        self.n_hidden = params.n_hidden
        self.batch_size = params.batch_size
        self.disable_memory = params.disable_memory
        self.summary_dir = params.summary_dir
        self.optimizer = params.optimizer
        self.learning_rate = params.learning_rate
        self.clip_gradients = params.clip_gradients
        self.data_dir = params.data_dir

        # Controller settings
        (self.n_train_instances, self.n_timesteps, self.n_env_inputs) = self.X_train.shape
        (self.n_test_instances, _, _) = self.X_test.shape
        (self.seq_output_len, self.n_classes)= self.y_train.shape
        self.n_read_inputs = self.W * self.R
        self.n_interface_outputs = (self.W * self.R) _ 3 * self.W + 5 * self.R + 3

        # Tensorflow settings
        self.session = tf.Session()
        # self.compile()
        if params.checkpoint_file:
            self.checkpoint_file_path = os.path.join('checkpoints', params.checkpoint_file)
            self.saver = tf.train.Saver()
            if os.path.exists(self.checkpoint_file_path):
                print('Restoring from checkpoint!')
                print()
                self.saver.restore(self.session, self.checkpoint_file_path)
            else:
                print('No checkpoint found! Starting from scratch...')
                print()

    def compile(self):
        self.read_keys_list = []
        self.write_keys_list = []
        self.allocation_gate_list = []
        self.free_gates_lits = []
        self.write_gate_list = []
        self.preds = []
        self.losses = []
        self.accuracies = []

        # Shortcuts
        N, W, R = self.N, self.W, self.R

        with tf.variable_scope('intput'):
            self.input_x = tf.placeholder(tf.float32, [self.batch_size, self.n_timesteps, self.n_env_inputs])
            self.input_y = tf.placeholder(tf.float32, [self.batch_size, self.seq_output_len, self.n_classes])

        self.memory = tf.fill([N, W], 1e-6, name='memory')

        # Read head variables


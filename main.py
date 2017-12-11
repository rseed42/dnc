#!/usr/bin/python3
import sys
import logging
import yaml
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Activation
# from keras.optimizers import RMSprop
# from keras.callbacks import TensorBoard
# from keras.utils.np_utils import to_categorical
#
# from dnc import DNC
import attrdict
from dataset import BabiDatasetLoader
# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
CONFIG_FILE = 'config.yml'
# SUMMARY_DIR = '/home/rseed42/Project/dnc/summary'
# VALIDATION_SPLIT = 0.25
# ------------------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------------------
log = logging.getLogger("main")
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
log.addHandler(handler)
# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
# class AttrDict(dict):
#     def __getattr__(self, key):
#         if key not in self:
#             raise AttributeError
#         return self[key]
#
#     def __setattr__(self, key, value):
#         if key not in self:
#             raise AttributeError
#         self[key] = value

# ------------------------------------------------------------------------------
# LSTM Model
# ------------------------------------------------------------------------------
#class LSTMModel:
#    def __init__(self, dataset):
#        # Easy
#        self.batch_size = 64
#        self.epoch_count = 10
#        self.learning_rate = 0.01
#        self.dataset = dataset
#        # Difficult
#        # self.batch_size = 1
#        # self.epoch_count = 100
#        # self.learning_rate = 0.0001
#
#        self.model = Sequential()
#        self.model.add(LSTM(512, input_shape=(self.dataset.longest_story_len, self.dataset.num_words)))
#        self.model.add(Dense(self.dataset.num_answers))
#        self.model.add(Activation('softmax'))
#
#        self.callbacks = [TensorBoard(log_dir='./logs')]
#        self.optimizer = RMSprop(lr=self.learning_rate)
#        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metric=['accuracy'])
#
#    def train(self):
#        # We need to convert the labels to one-hot encoded categorical data for Keras to understand
#        labels = to_categorical(self.dataset.Y_train)
#        validation_labels = to_categorical(self.dataset.Y_test)
#        self.model.fit(
#            self.dataset.X_train,
#            labels,
#            epochs=self.epoch_count,
#            batch_size=self.batch_size,
#            validation_data=(self.dataset.X_test, validation_labels),
#            callbacks=self.callbacks
#        )
#
# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == '__main__':
    # Here we go...
    log.debug('Loading configuration')

    # Step 01: Load configuration
    try:
        with open(CONFIG_FILE, 'r') as fp:
            config = attrdict.AttrDict(yaml.load(fp))
    except IOError:
        log.error('Could not load configuration file: {}'.format(CONFIG_FILE))
        sys.exit(1)

    # Step 02: Load the training and testing data
    try:
        data_loader = BabiDatasetLoader(config)
        data = data_loader.load(config.dataset.dir.cache, config.dataset.dir.data)
    except IOError:
        log.error('Failed to load the bAbI data set')
        sys.exit(1)

    # params = attrdict.AttrDict(
    #    N=1000,
    #    W=10,
    #    R=3,
    #    n_hidden=512,
    #    batch_size=1,
    #    disable_memory=False,
    #    summary_dir=SUMMARY_DIR,
    #    checkpoint_file=None,
    #    optimizer="RMSProp",
    #    learning_rate=0.001,
    #    clip_gradients=10.0,
    #    data_dir="./data"
    #
    # )

# machine = DNC(ds, params)

#!/usr/bin/python3
import os
import re
import glob
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation
from keras.optimizers import RMSprop
from keras.callbacks import TensorBoard
from keras.utils.np_utils import to_categorical

from dnc import DNC

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
DATA_DIR = '/home/rseed42/Data/babi/bAbi'
CACHE_DIR = '/home/rseed42/Project/dnc/cache'
SUMMARY_DIR = '/home/rseed42/Project/dnc/summary'
VALIDATION_SPLIT = 0.25

# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
class AttrDict(dict):
    def __getattr__(self, key):
        if key not in self:
            raise AttributeError
        return self[key]

    def __setattr__(self, key, value):
        if key not in self:
            raise AttributeError
        self[key] = value

# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------
class Dataset:

    def __init__(self):
        self.unique_words = set()
        self.unique_answers = set()
        self.stories = []
        self.longest_story_len = 0
        self.num_words = 0
        self.num_answers = 0
        self.X_train = None
        self.Y_train = None
        self.X_test = None
        self.Y_test = None

    def load(self, data_dir, cache_dir, clean=False):

        if os.path.exists(cache_dir):
            print('Loading data from cache')

            def load_array(filename):
                with open(os.path.join(cache_dir, filename), 'rb') as fp:
                    return np.load(fp)

            self.X_train = load_array('X_train')
            self.Y_train = load_array('Y_train')
            self.X_test = load_array('X_test')
            self.Y_test = load_array('Y_test')
            # Fix later
            # [load_array(fn, ar) for fn, ar in [
            #     ('X_train', self.X_train),
            #     ('Y_train', self.Y_train),
            #     ('X_test', self.X_test),
            #     ('Y_test', self.Y_test)
            # ]]
            return

        en10k_dir = os.path.join(data_dir, 'en-10k/*')
        for filename in glob.glob(en10k_dir):
            print (filename)
            # stories = []
            with open(filename, 'r') as fp:
                # Read all lines and remove any digits
                lines = [str(re.sub("\d", "", line)).strip() for line in fp.readlines()]
                story = []
                for line in lines:
                    line = line.replace(",", " , ").replace("?", " ? ")
                    line = line.replace(".", " . ").replace(",", " , ").replace('\'', '')
                    # Not a question
                    if "\t" not in line:
                        words = line.split()
                        self.unique_words = self.unique_words.union(set(words))
                        story.extend(words)
                    # Question
                    else:
                        [line, answer] = line.split("\t")
                        words = line.split()
                        self.unique_words = self.unique_words.union(set(words))
                        self.unique_answers = self.unique_answers.union(set([answer]))
                        story.extend(words)
                        this_story = {"seq": story, "answer": answer}
                        self.stories.append(this_story)
                        story = []
                break
        self.process(VALIDATION_SPLIT)
        self.store_cache(cache_dir)


    def store_cache(self, cache_dir):
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)

        def save_to_file(filename, array):
            with open(os.path.join(cache_dir, filename), 'wb') as fp:
                np.save(fp, array)

        [save_to_file(fn, ar) for fn, ar in (
            ('X_train', self.X_train),
            ('Y_train', self.Y_train),
            ('X_test', self.X_test),
            ('Y_test', self.Y_test)
        )]

    def pad_and_encode_seq(self, encoder, seq, seq_len):
        if len(seq) > seq_len:
            raise RuntimeError('Should never see a sequence greater  than {} length'.format(seq_len))
        return encoder.transform((['' for _ in range(seq_len - len(seq))]) + seq)

    def process(self, validation_split):
        longest_story = max(self.stories, key=lambda s: len(s['seq']))
        self.longest_story_len = len(longest_story['seq'])
        print(
            'Input will be a sequence of {} words, ' \
            'padded by zeroes at the beginning when ' \
            'needed.'.format(len(longest_story['seq']))

        )
        self.num_words = len(self.unique_words)
        self.num_answers = len(self.unique_answers)
        print('There are {} unique words, which will be mapped to one-hot encoded vectors.'.format(self.num_words))
        print('There are {} unique answers, which will be mapped to one-hot encoded vectors.'.format(self.num_answers))

        # Create the one-hot encoded word labels
        lb = preprocessing.LabelBinarizer()
        word_encoder = lb.fit(list(self.unique_words))

        # Create the one-hot encoded answer labels
        lb = preprocessing.LabelBinarizer()
        answer_encoder = lb.fit(list(self.unique_answers))

        # Encode
        print()
        print('Encoding sequences...')
        X = []
        Y = []

        for story in self.stories:
            X.append(np.array(self.pad_and_encode_seq(word_encoder, story['seq'], self.longest_story_len)))
            Y.append(answer_encoder.transform([story["answer"]])[0])
            # Y.append(answer_encoder.transform([story["answer"]]))

        print('Splitting training/test set...')
        X = np.array(X)
        Y = np.array(Y)

        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=validation_split)
        print()
        print('X_train: ', self.X_train.shape)
        print('Y_train: ', self.Y_train.shape)
        print('X_test: ', self.X_test.shape)
        print('Y_test: ', self.Y_test.shape)

# ------------------------------------------------------------------------------
# LSTM Model
# ------------------------------------------------------------------------------
class LSTMModel:
    def __init__(self, dataset):
        # Easy
        self.batch_size = 64
        self.epoch_count = 10
        self.learning_rate = 0.01
        self.dataset = dataset
        # Difficult
        # self.batch_size = 1
        # self.epoch_count = 100
        # self.learning_rate = 0.0001

        self.model = Sequential()
        self.model.add(LSTM(512, input_shape=(self.dataset.longest_story_len, self.dataset.num_words)))
        self.model.add(Dense(self.dataset.num_answers))
        self.model.add(Activation('softmax'))

        self.callbacks = [TensorBoard(log_dir='./logs')]
        self.optimizer = RMSprop(lr=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=self.optimizer, metric=['accuracy'])

    def train(self):
        # We need to convert the labels to one-hot encoded categorical data for Keras to understand
        labels = to_categorical(self.dataset.Y_train)
        validation_labels = to_categorical(self.dataset.Y_test)
        self.model.fit(
            self.dataset.X_train,
            labels,
            epochs=self.epoch_count,
            batch_size=self.batch_size,
            validation_data=(self.dataset.X_test, validation_labels),
            callbacks=self.callbacks
        )

# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
if __name__ == '__main__':

    ds = Dataset()
    ds.load(DATA_DIR, CACHE_DIR)

#    params = AttrDict(
#        N=ds.X_train.shape[1],
#        W=10,
#        R=3,
#        n_hidden=512,
#        batch_size=1,
#        disable_memory=False,
#        summary_dir=SUMMARY_DIR,
#        checkpoint_file=None,
#        optimizer="RMSProp",
#        learning_rate=0.001,
#        clip_gradients=10.0,
#        data_dir="./data"
#
#    )
#
# machine = DNC(ds, params)

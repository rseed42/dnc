import sys
import os
import re
import glob
import numpy as np
# from sklearn import preprocessing
# from sklearn.cross_validation import train_test_split
import logging
from functools import reduce
# ------------------------------------------------------------------------------
# Globals
# ------------------------------------------------------------------------------
# Let's spare us some typing
log = logging.getLogger("dataset")
log.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
log.addHandler(handler)
# ------------------------------------------------------------------------------
# Errors
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# bAbI data set loader and data object
# ------------------------------------------------------------------------------


class DatasetBabi:
    def __init__(self, training_data, training_labels, testing_data, testing_labels):
        self.training_data = training_data
        self.training_labels = training_labels
        self.testing_data = testing_data
        self.testing_labels = testing_labels


class StorySet:
    def __init__(self, unique_words, unique_answers, stories):
        self.unique_words = unique_words
        self.unique_answers = unique_answers
        self.stories = stories
        self.unique_words_len = len(unique_words)
        self.unique_answers_len = len(unique_answers)


class Story:
    def __init__(self):
        self.words = []
        self.answer = ''
        self.unique_words = set()

    def update_words(self, words):
        self.words.extend(words)
        self.unique_words = self.unique_words.union(set(words))

    def __str__(self):
        return '{}: {} : {}'.format(' '.join(self.words), self.answer, len(self.unique_words))

# ------------------------------------------------------------------------------
# Processing functions grouped together as static methods for convenience
# ------------------------------------------------------------------------------


class Preprocessor:
    @staticmethod
    def process(line):
        padded_line = Preprocessor.pad_punctuation_marks(line)
        return Preprocessor.strip_digit(padded_line)

    @staticmethod
    def pad_punctuation_marks(line):
        return line \
            .replace(",", " , ") \
            .replace("?", " ? ") \
            .replace(".", " . ") \
            .replace(",", " , ") \
            .replace('\'', '')

    @staticmethod
    def strip_digit(line):
        return str(re.sub('\d', '', line)).strip()


class FileProcessor:
    @staticmethod
    def stories_generator(lines):
        story = Story()
        for line in lines:
            # If the line contains a tab, then it is a question-answer pair. Otherwise, it is a statement
            line_parts = line.split('\t')
            # The first part of the line is always either a statement or a question. Update the story with
            # the words that it contains
            story.update_words(line_parts[0].split())

            if len(line_parts) > 1:
                story.answer = line_parts[1]
                yield story
                story = Story()

    @staticmethod
    def calculate_unique(prev, story):
        return prev[0].union(story.unique_words), prev[1].union([story.answer])

    @staticmethod
    def process(filename):
        with open(filename, 'r') as fp:
            lines = map(Preprocessor.process, fp)
            stories = FileProcessor.stories_generator(lines)
            # Reduce is a terminal operation and forces all transformations to be applied, therefore
            # the whole file will be processed when we go out of scope
            unique_words, unique_answers = reduce(FileProcessor.calculate_unique, stories, (set(), set()))
            return StorySet(unique_words, unique_answers, stories)


class OneHotEncoder:
    @staticmethod
    def encode(story_set):
        log.info('Unique words: {}'.format(story_set.unique_words_len))
        return story_set

class BabiDatasetLoader:
    def __init__(self, config):
        self.config = config
        self.unique_words = set()
        self.unique_answers = set()
        # self.stories = []
        self.longest_story_len = 0
        self.unique_word_count = 0
        self.unique_answer_count = 0

    def load(self, cache_dir, data_dir):
        """
        Load the data from cache if available. Otherwise, process and store it first, then load it.
        :param cache_dir:
        :param data_dir:
        :return:
        """
        log.debug("Trying to load cached data from: {}".format(cache_dir))
        if not os.path.exists(cache_dir):
            log.info("No cached data found. Processing dataset...")
            self.process_and_store_data(data_dir, cache_dir)
        return self.load_from_cache(cache_dir)

    def load_from_cache(self, cache_dir):
        """
        Read the numpy arrays from the cache dir
        :param cache_dir:
        :return:
        """
        # An IO Exception is caught in the main function
        def load_array(filename):
            with open(os.path.join(cache_dir, filename), 'rb') as fp:
                return np.load(fp)

        return DatasetBabi(
             load_array(self.config.dataset.name.train.data),
             load_array(self.config.dataset.name.train.labels),
             load_array(self.config.dataset.name.test.data),
             load_array(self.config.dataset.name.test.labels)
        )

    def process_data_file(self, filename):
        """
        Stream processing of the data
        :param filename:
        :return:
        """
        log.info('Processing: {}'.format(filename))
        with open(filename, 'r') as fp:
            # Clean up the input first
            clean_lines = map(self.process_line, fp)
            # Convert the lines to stories
            return list(self.gen_stories(clean_lines))

    def process_and_store_data(self, data_dir, cache_dir):
        log.debug("Processing data")
        glob_en10k_dir_all_files = os.path.join(data_dir, 'en-10k/*')

        # Do not process everything during development
        whitelist = ('/home/rseed42/Data/babi/bAbi/en-10k/qa1_single-supporting-fact_train.txt',)
        # 01. Create a stream of files to be processed
        data_files = filter(lambda s: s in whitelist, glob.glob(glob_en10k_dir_all_files))
        # 02. Convert each file into a sequence of stories
        story_sets = map(FileProcessor.process, data_files)

        # 03. One-hot encode the individual story sets
        encoded_sets = map(OneHotEncoder.encode, story_sets)

        # 04. Save the data sets

        # 05. Count the number of story sets as a terminal operation
        story_set_count = reduce(lambda p, s: p + 1, encoded_sets, 0)

        log.info('Processed {} story sets'.format(story_set_count))

        return story_set_count

        # self.process(VALIDATION_SPLIT)
        # self.store_cache(cache_dir)

    def one_hot_encode(self, story_set):
        pass




    #
    #
    # def store_cache(self, cache_dir):
    #     if not os.path.exists(cache_dir):
    #         os.mkdir(cache_dir)
    #
    #     def save_to_file(filename, array):
    #         with open(os.path.join(cache_dir, filename), 'wb') as fp:
    #             np.save(fp, array)
    #
    #     [save_to_file(fn, ar) for fn, ar in (
    #         ('X_train', self.X_train),
    #         ('Y_train', self.Y_train),
    #         ('X_test', self.X_test),
    #         ('Y_test', self.Y_test)
    #     )]
    #
    # def pad_and_encode_seq(self, encoder, seq, seq_len):
    #     if len(seq) > seq_len:
    #         raise RuntimeError('Should never see a sequence greater  than {} length'.format(seq_len))
    #     return encoder.transform((['' for _ in range(seq_len - len(seq))]) + seq)
    #
    # def process(self, validation_split):
    #     longest_story = max(self.stories, key=lambda s: len(s['seq']))
    #     self.longest_story_len = len(longest_story['seq'])
    #     print(
    #         'Input will be a sequence of {} words, ' \
    #         'padded by zeroes at the beginning when ' \
    #         'needed.'.format(len(longest_story['seq']))
    #
    #     )
    #     self.num_words = len(self.unique_words)
    #     self.num_answers = len(self.unique_answers)
    #     print('There are {} unique words, which will be mapped to one-hot encoded vectors.'.format(self.num_words))
    #     print('There are {} unique answers, which will be mapped to one-hot encoded vectors.'.format(self.num_answers))
    #
    #     # Create the one-hot encoded word labels
    #     lb = preprocessing.LabelBinarizer()
    #     word_encoder = lb.fit(list(self.unique_words))
    #
    #     # Create the one-hot encoded answer labels
    #     lb = preprocessing.LabelBinarizer()
    #     answer_encoder = lb.fit(list(self.unique_answers))
    #
    #     # Encode
    #     print()
    #     print('Encoding sequences...')
    #     X = []
    #     Y = []
    #
    #     for story in self.stories:
    #         X.append(np.array(self.pad_and_encode_seq(word_encoder, story['seq'], self.longest_story_len)))
    #         Y.append(answer_encoder.transform([story["answer"]])[0])
    #         # Y.append(answer_encoder.transform([story["answer"]]))
    #
    #     print('Splitting training/test set...')
    #     X = np.array(X)
    #     Y = np.array(Y)
    #
    #     self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y, test_size=validation_split)
    #     print()
    #     print('X_train: ', self.X_train.shape)
    #     print('Y_train: ', self.Y_train.shape)
    #     print('X_test: ', self.X_test.shape)
    #     print('Y_test: ', self.Y_test.shape)


import sys
import os
import re
import numpy as np
from sklearn import preprocessing
import logging
from functional import seq
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
# bAbI data set loader and data object
# ------------------------------------------------------------------------------


class DatasetBabi:
    """
    """
    def __init__(self, name, training_data, training_labels, testing_data, testing_labels):
        self.name = name
        self.training_data = training_data
        self.training_labels = training_labels
        self.testing_data = testing_data
        self.testing_labels = testing_labels


class StorySetInfo:
    """
    Data structure that is used to determine the statistics for the set of stories
    """
    def __init__(self, longest_story: int, unique_words: set, unique_answers: set):
        """
        Construct the info object
        :param longest_story:  int
        :param unique_words:   set
        :param unique_answers:  set
        """
        self.longest_story = longest_story
        self.unique_words = unique_words
        self.unique_answers = unique_answers
        self.unique_words_len = len(self.unique_words)
        self.unique_answers_len = len(self.unique_answers)

    def __str__(self) -> str:
        return 'max story len: {} unique words len: {} unique answers len: {}'.format(
            self.longest_story,
            self.unique_words_len,
            self.unique_answers_len
        )

    @staticmethod
    def update(info, story):
        return StorySetInfo(
            max(info.longest_story, len(story.words)),
            info.unique_words.union(set(story.words)),
            info.unique_answers.union(set([story.answer]))
        )


class Story:
    def __init__(self, words, answer):
        self.words = words
        self.answer = answer

    def __str__(self):
        return '{}: {}'.format(' '.join(self.words), self.answer)

# ------------------------------------------------------------------------------
# Processing functions grouped together as static methods for convenience
# ------------------------------------------------------------------------------


class BabiDatasetLoader:

    @staticmethod
    def load(parent_cache_dir, data_dir, dataset):
        """
        Load the data from cache if available. Otherwise, process and store it first, then load it.
        :param parent_cache_dir: The cache dir where all different data sets are placed as subdirectories
        :param data_dir: The data directory containing the bAbi dataset
        :param dataset: Dataset configuration
        :return:
        """
        cache_dir = os.path.join(parent_cache_dir, dataset.name)
        log.debug("Trying to load cached data from: {}".format(cache_dir))
        if not os.path.exists(cache_dir):
            log.info('No cached data found. Processing dataset {}'.format(dataset.name))
            log.info('Creating cache directory: {}'.format(cache_dir))
            try:
                os.makedirs(cache_dir)
            except IOError:
                log.error('Could not create cache directory')
                return
            BabiDatasetLoader.process_and_store_data(
                os.path.join(data_dir, dataset.train.data_file),
                os.path.join(cache_dir, dataset.train.train_cache_file),
                os.path.join(cache_dir, dataset.train.label_cache_file)
            )
            log.info('Processing testing data: {}'.format(dataset.test.data_file))
            BabiDatasetLoader.process_and_store_data(
                os.path.join(data_dir, dataset.test.data_file),
                os.path.join(cache_dir, dataset.test.test_cache_file),
                os.path.join(cache_dir, dataset.test.label_cache_file)
            )

        return BabiDatasetLoader.load_from_cache(
            dataset.name,
            cache_dir,
            dataset.train.train_cache_file,
            dataset.train.label_cache_file,
            dataset.test.test_cache_file,
            dataset.test.label_cache_file
        )

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

    @staticmethod
    def load_from_cache(dataset_name, cache_dir, train_file, train_labels_file, test_file, test_labels_file):
        """
        Read the numpy arrays from the cache dir
        :param dataset_name:
        :param cache_dir:
        :param train_file:
        :param train_labels_file:
        :param test_file:
        :param test_labels_file:
        :return:
        """
        # An IO Exception is caught in the main function
        def load_array(filename):
            with open(os.path.join(cache_dir, filename), 'rb') as fp:
                ar = np.load(fp, allow_pickle=False)
                log.info('Loading array shape: {} from file: {}'.format(ar.shape, filename))
                return ar

        return DatasetBabi(
             dataset_name,
             load_array(train_file),
             load_array(train_labels_file),
             load_array(test_file),
             load_array(test_labels_file)
        )

    @staticmethod
    def create_stories(current, next_line):
        line_parts = next_line.split('\t')
        # If we split by tab and have two elements, then the second element is the answer. In this case,
        # we need to create the story object from the accumulated lines so far.
        if len(line_parts) > 1:
            # Concatenate all statements and then split by space
            str_words = ' '.join(current[-1]) + line_parts[0]
            # Construct the story and replace the previous list of lines that comprise it
            return current[:-1] + [Story(str_words.split(), line_parts[1])] + [[]]
        return current[:-1] + [current[-1] + [next_line]]

    @staticmethod
    def process_and_store_data(data_file, data_cache_file, label_cache_file):
        log.info('Processing {}'.format(data_file))

        # Encode the stories
        def encode_story(story):
            padded_words = [''] * info.longest_story
            # Put the words to the right of the padded area (TODO: Check if this makes sense and use the left instead)
            padded_words[info.longest_story - len(story.words):] = story.words
            return word_encoder.transform(padded_words)

        # 1. Read the data file line by line.
        # 2. Pre-process each line
        # 3. Fold contiguous blocks of lines into stories (end detected by the presence of an answer)
        # 4. Remove the last empty array that results in from building up the stories
        stories = seq\
            .open(data_file)\
            .map(BabiDatasetLoader.strip_digit)\
            .map(BabiDatasetLoader.pad_punctuation_marks)\
            .fold_left([[]], BabiDatasetLoader.create_stories)\
            .filter(lambda s: isinstance(s, Story))

        # 5. Find out the maximum story length, the unique words, and the number of unique answers
        info = stories.reduce(StorySetInfo.update, StorySetInfo(0, set(), set()))
        log.info('Stories summary: {}'.format(info))
        # Create the one-hot encoded word labels
        label_binarizer = preprocessing.LabelBinarizer()
        word_encoder = label_binarizer.fit(list(info.unique_words))
        # Create the one-hot encoded answer labels
        label_binarizer = preprocessing.LabelBinarizer()
        answer_encoder = label_binarizer.fit(list(info.unique_answers))
        # 6. One-hot-encode the data
        one_hot_encoded_stories = stories.map(encode_story)
        # 7. One-hot-encode the answers (labels)
        one_hot_encoded_answers = stories.map(lambda story: answer_encoder.transform([story.answer])[0])
        # Store the data in the cache dir
        log.info('Storing data cache file {}'.format(data_cache_file))
        BabiDatasetLoader.store_file(data_cache_file, one_hot_encoded_stories)
        # Store the answers in the cache dir
        log.info('Storing label cache file {}'.format(label_cache_file))
        BabiDatasetLoader.store_file(label_cache_file, one_hot_encoded_answers)

    @staticmethod
    def store_file(filename, sequence):
        with open(filename, 'wb') as fp:
            one_hot_encoded_array = np.array(sequence.to_list())
            log.info('Storing one hot encoded array {}'.format(one_hot_encoded_array.shape))
            np.save(fp, one_hot_encoded_array, allow_pickle=False)

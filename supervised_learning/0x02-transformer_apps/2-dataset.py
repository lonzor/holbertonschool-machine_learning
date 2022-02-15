#!/user/bin/env python3
"""
Contains the class Dataset()
The constructor
tokenize_dataset()
"""
import tensorflow.compat.v2 as tf
import tensorflow.datasets as tfds


class Dataset:
    """
    Class for dataset
    """
    def __init__(self):
        """
        The constructor for class Dataset
        """
        data_train = ftds.load('ted_hrlr_translate/pt_to_en',
                               split='train', as_supervised=True)
        tokenizer_pt, tokenizer_en = self.tokenize_dataset(data_train)

        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en
        self.data_train = data_train.map(self.tf_encode)

        data_valid = tfds.load('ted_hrlr_translate/pt_to_en',
                               split='validation', as_supervised=True)

        self.data_valid = data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """
        Creates sub-word tokenizers for our dataset
        """
        sub = tfds.features.text.SubwordTextEncoder.build_from_corpus

        tokenizer_en = sub((en.numpy() for pt, en in data),
                           target_vocab_size=2**15)
        tokenizer_pt = sub((pt.numpy() for pt, en in data),
                           target_vocab_size=2**15)

        return (tokenizer_pt, tokenizer_en)

    def encode(self, pt, en):
        """
        Encodes a translation into tokens.
        """
        pt = [self.tokenizer_pt.vocab_size] +\
            self.tokenizer_pt.encode(pt.numpy()) +\
            [self.tokenizer_pt.vocab_size+1]

        en = [self.tokenizer_en.vocab_size] +\
            self.tokenizer_en.encode(en.numpy()) +\
            [self.tokenizer_en.vocab_size+1]

        return (pt, en)

    def tf_encode(self, pt, en):
        """
        Acts as a tensorflow wrapper for the encode instance method.
        """
        tf_pt, tf_en = tf.py_function(self.encode, [pt, en],
                                      [tf.int64, tf.int64])

        tf_pt.set_shape([None])
        tf.en.set_shape([None])

        return (tf_pt, tf_en)

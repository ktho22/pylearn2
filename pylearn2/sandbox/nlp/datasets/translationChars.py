"""
Dataset wrapper for Google's pre-trained word2vec embeddings.
This dataset maps sequences of character indices to word embeddings.

See: https://code.google.com/p/word2vec/
"""
import cPickle
from functools import wraps

import numpy as np
import tables

from pylearn2.sandbox.nlp.datasets.text import TextDatasetMixin
from pylearn2.sandbox.rnn.space import SequenceDataSpace
from pylearn2.sandbox.rnn.utils.iteration import SequenceDatasetIterator
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.space import IndexSpace, CompositeSpace, VectorSpace
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils.string_utils import preprocess


class TranslationChars(VectorSpacesDataset, TextDatasetMixin):
    """
    Loads the data from a PyTables VLArray (character indices)
    and CArray (word embeddings) and stores them as an array
    of arrays and a matrix respectively.

    Parameters
    ----------
    which_set : str
        Either `train` or `valid`
    """
    def __init__(self, which_set, stop=None, start=0):
        assert which_set in ['train', 'valid', 'train_cost']
        self._stop = stop
        self._start = start
        # TextDatasetMixin parameters
        self._unknown_index = 0
        self._case_sensitive = True

        valid_start = 10000
        train_cost_start = 11000
        train_cost_stop = 12000

        # if which_set == 'train':
        #     assert (stop == None or stop <= 28000), "There are only 29000 words in training set"
        #     self._stop = stop
        #     if stop is None:
        #         self._stop = 28000
        # if which_set == 'valid':
        #     assert (stop == None or stop <= 2000), "There are only 1000 words in validation set"
        #     if stop is None:
        #         self._stop = 30000
        #     else: 
        #         self._stop = stop + 28000
        #     self._start = 29000

        with open('/data/lisatmp3/devincol/data/translation_char_vocab.en.pkl') as f:
            self._vocabulary = cPickle.load(f)

        with open('/data/lisatmp3/devincol/data/translation_vocab_aschar.en.pkl') as f:
            raw = cPickle.load(f)
            if which_set == 'train':
                raw1 = raw[:valid_start]
                print "raw1", raw1.shape
                raw2 = raw[train_cost_stop:]
                raw = np.concatenate((raw1, raw2))
            elif which_set == 'valid':
                raw = raw[valid_start:train_cost_start]
            else:
                raw = raw[train_cost_start:train_cost_stop]
            self.X = np.asarray([char_sequence[:, np.newaxis]
                                 for char_sequence in raw])
        print "X shape", self.X.shape

        # Load the data
        print "loading embeddings"
        
        raw = np.load('/data/lisatmp3/chokyun/emb.npy')
        if which_set == 'train':
            raw1 = raw[:valid_start]
            print "raw1 shape", raw1.shape
            raw2 = raw[train_cost_stop:-1]
            self.y = np.concatenate((raw1, raw2))
        elif which_set == 'valid':
            self.y = raw[valid_start:train_cost_start]
        else:
            self.y = raw[train_cost_start:train_cost_stop]
        
        print "y shape", self.y.shape

        source = ('features', 'targets')
        space = CompositeSpace([
            SequenceDataSpace(IndexSpace(dim=1, max_labels=144)),
            VectorSpace(dim=620)])
        super(TranslationChars, self).__init__(data=(self.X, self.y),
                                       data_specs=(space, source))

    def _create_subset_iterator(self, mode, batch_size=None, num_batches=None,
                                rng=None):
        subset_iterator = resolve_iterator_class(mode)
        if rng is None and subset_iterator.stochastic:
            rng = make_np_rng()
        return subset_iterator(self.get_num_examples(), batch_size,
                               num_batches, rng)

    @wraps(VectorSpacesDataset.iterator)
    def iterator(self, batch_size=None, num_batches=None, rng=None,
                 data_specs=None, return_tuple=False, mode=None):
        subset_iterator = self._create_subset_iterator(
            mode=mode, batch_size=batch_size, num_batches=num_batches, rng=rng
        )
        return SequenceDatasetIterator(self, data_specs, subset_iterator,
                                       return_tuple=return_tuple)


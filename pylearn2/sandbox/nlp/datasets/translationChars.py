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
    def __init__(self, which_set, stop=None):
        assert which_set in ['train', 'valid']
        self._stop = stop
        # TextDatasetMixin parameters
        self._unknown_index = 0
        self._case_sensitive = True
        with open(preprocess('/data/lisatmp3/devincol/data/translation_vocab_aschar.en.pkl')) as f:
            self.X = cPickle.load(f)

        # Load the data
        print "loading embeddings"
        with tables.open_file(preprocess('/data/lisatmp3/chokyun/emb.npy')) as f:
            self.y = np.load(f)
            

        source = ('features', 'targets')
        space = CompositeSpace([SequenceDataSpace(IndexSpace(dim=1,
                                                             max_labels=101)),
                                VectorSpace(dim=300)])
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


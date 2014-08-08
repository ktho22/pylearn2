"""
Pylearn2 wrapper for h5-format datasets of sentences. Dataset generates
ngrams and swaps 2 adjacent words. Targets are n-1 vectors indicating where 
swap happened. 
"""
__authors__ = ["Coline Devin"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Coline Devin"]
__license__ = "3-clause BSD"
__maintainer__ = "Coline Devin"
__email__ = "devincol@iro"


import os.path
import functools
import numpy
import tables
import cPickle
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dataset import Dataset
from pylearn2.sandbox.nlp.datasets.shuffle2 import H5Shuffle
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace, Conv2DSpace
from pylearn2.utils import serial
from pylearn2.utils import safe_zip
from pylearn2.utils.iteration import FiniteDatasetIterator
from pylearn2.sandbox.rnn.space import SequenceDataSpace
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.utils.rng import make_np_rng
from pylearn2.sandbox.rnn.utils.iteration import SequenceDatasetIterator

def index_from_one_hot(one_hot):
    return numpy.where(one_hot == 1.0)[0][0]

class H5RnnSkipgram(H5Shuffle):
    """
    Frame-based WMT14 dataset
    """
    _default_seed = (17, 2, 946)

    def __init__(self, path, node, which_set, frame_length,
                 start=0, stop=None, X_labels=None,
		 _iter_num_batches=None, rng=_default_seed, 
                 load_to_memory=False, cache_size=None,
                 cache_delta=None):
        """
        Parameters
        ----------
        path : str
            The base path to the data
        node: str
            The node in the h5 file
        which_set : str
            Either "train", "valid" or "test"
        frame_length : int
            Number of words contained in a frame
        start : int, optional
            Starting index of the sequences to use. Defaults to 0.
        stop : int, optional
            Ending index of the sequences to use. Defaults to `None`, meaning
            sequences are selected all the way to the end of the array.
        rng : object, optional
            A random number generator used for picking random indices into the
            design matrix when choosing minibatches.
        """
        print "enter"
        super(H5RnnSkipgram, self).__init__(path, node, which_set, frame_length,
                 start=start, stop=stop, X_labels=X_labels,
		 _iter_num_batches=_iter_num_batches, rng=rng, 
                 load_to_memory=load_to_memory, cache_size=cache_size,
                 cache_delta=cache_delta)

        self._load_dicts()
        features_space = SequenceDataSpace(IndexSpace(dim=1, max_labels=213))
        features_source = 'features'

        targets_space = [IndexSpace(dim=1, max_labels=self.X_labels)]
        targets_source = 'targets'

        spaces = [features_space, targets_space]
        space = CompositeSpace(spaces)
        source = (features_source, targets_source)
        self.data_specs = (space, source)

        def getFeatures(indexes):
            """
            .. todo::
                Write me
            """
            if self._load_to_memory:
                sequences = [self.samples_sequences[i] for i in indexes]
            else:
                sequences = [self.node[i] for i in indexes]
            # Get random start point for ngram
            source_i = [numpy.random.randint(self.frame_length/2 +1, len(s)-self.frame_length/2, 1)[0]
                        for s in sequences]
            target_i = [min(abs(int(numpy.random.normal(s_i, self.frame_length/3.0))), len(s)-1)
                        for s_i, s in safe_izip(source_i, sequences)]

            def make_sequence(word):
                string = self._inv_words[word]
                seq = map(lambda c: [self._char_labels[c]], self._inv_words[word])
                seq.append([self._eow])
                return numpy.asarray(seq)

            X = [numpy.asarray([s[make_sequence(i)]]) for i, s in safe_izip(source_i, sequences)]
            X[X>=self.X_labels] = numpy.asarray([1])
            X = numpy.asarray(X)
            y = [numpy.asarray([s[i]]) for i, s in safe_izip(target_i, sequences)]
            y[y>=self.X_labels] = numpy.asarray([1])
            y = numpy.asarray(y)
            self.lastY = (y, indexes)
            return X

        def getTarget(source_index, indexes):
            if numpy.array_equal(indexes, self.lastY[1]):
                #y = numpy.transpose(self.lastY[0][:,source_index][numpy.newaxis])
                return self.lastY[0]
            else:
                print "You can only ask for targets immediately after asking for those features"
                return None
               
        self.sourceFNs['target'] = getTarget
        self.sourceFNs['features'] = getFeatures
        
    def _load_dicts(self):
        word_dict_path = "/data/lisatmp3/pougetj/vocab.pkl"
        char_dict_path = "/data/lisatmp3/pougetj/char_vocab.pkl"
        with open(word_dict_path) as f:
            word_labels = cPickle.load(f)
        self._inv_words = {v:k for k, v in word_labels.items()}
        with open(char_dict_path) as f:
            self._char_labels = cPickle.load(f)
        self._eow = len(self._char_labels)

    def _create_subset_iterator(self, mode, batch_size=None, num_batches=None,
                                rng=None):
        subset_iterator = resolve_iterator_class(mode)
        if rng is None and subset_iterator.stochastic:
            rng = make_np_rng()
        return subset_iterator(self.get_num_examples(), batch_size,
                               num_batches, rng)

    @functools.wraps(Dataset.iterator)
    def iterator(self, batch_size=None, num_batches=None, rng=None,
                 data_specs=None, return_tuple=False, mode=None):
        subset_iterator = self._create_subset_iterator(
            mode=mode, batch_size=batch_size, num_batches=num_batches, rng=rng
        )
        return SequenceDatasetIterator(self, data_specs, subset_iterator,
                                       return_tuple=return_tuple)


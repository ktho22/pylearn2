"""
Pylearn2 wrapper for h5-format datasets of sentences. Dataset generates
ngrams and swaps 2 adjacent words. Targets are n-1 vectors indicating where 
swap happened. 
"""
__authors__ = ["Coline Devin"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = ["Coline Devin", "Vincent Dumoulin"]
__license__ = "3-clause BSD"
__maintainer__ = "Coline Devin"
__email__ = "devincol@iro"


import os.path
import functools
import numpy
import tables
from pylearn2.utils.iteration import resolve_iterator_class
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pylearn2.datasets.dataset import Dataset
from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace, Conv2DSpace
from pylearn2.utils import serial
from pylearn2.utils import safe_zip
from pylearn2.utils.iteration import FiniteDatasetIterator
from multiprocessing import Process, Queue
import ipdb
from shuffle2 import H5Shuffle

def index_from_one_hot(one_hot):
    return numpy.where(one_hot == 1.0)[0][0]

class H5Shuffle(H5Shuffle):
    """
    Frame-based WMT14 dataset
    """
    _default_seed = (17, 2, 946)

    def __init__(self, path, node, which_set, frame_length,
                 start=0, stop=None, X_labels=None,
		 _iter_num_batches=None, rng=_default_seed, 
                 load_to_memory=False, cache_size=None,
                 cache_delta=None, schwenk=False):
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
	_iter_num_batches : int, optional
	    Determines number of batches to cycle through in one epoch. Used to
	    calculate the validation score without having to cycle through the
	    entire dataset
        load_to_memory : bool, optional
            If True, will load all requested data into memory. This allows the
            iterations to go faster, but requires significantly more memory.
        cache_size : int, optional
            If cache_size is set, the dataset will initially only load the 
            first cache_size examples from the data. Making this larger will
            increase the possible distance between examples (because data is
            loaded sequentially)
        cache_delta : int, optional
            Required if cache_size is set. Every cache_delta examples 
            (approximately because it doesn't need to be a multiple of batches)
            the dataset will load an additional cache_delta examples from the
            data (in consecutive order). Making this larger will allow more 
            non-sequentiality, but if cache_delta is equal to cache_size,
            then there will be no overlap between caches.
        """
        self.base_path = path
        self.node_name = node
        self.which_set = which_set
        self.frame_length = frame_length
        self.X_labels = X_labels
       
        if type(stop)==str:
            self._stop = eval(stop)
        else:
            self._stop = stop
        self._start = start
	if _iter_num_batches is None:
		self._iter_num_batches = 1000
	else:
		self._iter_num_batches = _iter_num_batches
        self._load_to_memory = load_to_memory
        self.schwenk = schwenk
        self._using_cache = False
        #self.y_labels = y_labels
        if cache_size is not None:
            assert cache_delta is not None, (
                "cache_delta cannot be None if cache_size is set"
            )
            assert cache_size >= cache_delta, (
                "cache_delta must be less than or equal to cache_size"
            )
            self._using_cache = True
            self._cache_size = cache_size
            self._cache_delta = cache_delta
            self._data_queue = Queue()
            self._num_since_last_load = 0
            self._next_cache_index = cache_delta + cache_size + start
            self._loading = False

        # RNG initialization
        if hasattr(rng, 'random_integers'):
            self.rng = rng
        else:
            self.rng = numpy.random.RandomState(rng)


        if self._using_cache:
            self._load_data(which_set, (start, start+cache_size))
        else:
            self._load_data(which_set, (start, stop))
            
        # self.cumulative_sequence_indexes = numpy.cumsum(len(s) for s in self.raw_data)
  
        # DataSpecs
        features_space = IndexSpace(
            dim=self.frame_length,
            max_labels=self.X_labels
        )
        features_source = 'features'

        targets_space = VectorSpace(dim=self.frame_length-1)
        targets_source = 'targets'
        # def targets_map_fn(indexes):

        space = CompositeSpace([features_space, targets_space])
        source = (features_source, targets_source)

        self.data_specs = (space, source)

        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('shuffled_sequential')
        self._iter_data_specs = (CompositeSpace((features_space,
                                                 targets_space)),
                                 (features_source, targets_source))
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
            wis = [numpy.random.randint(0, len(s)-self.frame_length+1) for s in sequences]
            X = numpy.asarray([s[wi:self.frame_length+wi] for s, wi in zip(sequences, wis)])

            # Words mapped to integers greater than input max are set to 1 (unknown)
            X[X>=self.X_labels] = 1
            
            replace_idx = numpy.random.randint(0, self.frame_length-1, len(X))
            replaced_word_idx = numpy.random.randint(0,self.X_labels,1)
            X[0,replace_idx] = replaced_word_idx

            y = numpy.zeros((len(X), self.frame_length - 1))
            y[numpy.arange(len(X)), replace_idx] = 1

            # Store the targets generated by these indices.
            self.lastY = (y, indexes)
            return X

        def getTarget(indexes):
            if numpy.array_equal(indexes, self.lastY[1]):
                return self.lastY[0]
            else:
                return None
        self.sourceFNs = {'features': getFeatures, 'targets': getTarget}


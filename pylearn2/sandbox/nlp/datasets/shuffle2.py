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

def index_from_one_hot(one_hot):
    return numpy.where(one_hot == 1.0)[0][0]

class H5Shuffle(Dataset):
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
            self._max_data_index = stop
            self._start = start
            self._data_queue = Queue()
            self._num_since_last_load = 0
            self._next_cache_index = cache_delta + cache_size + start
            self._loading = False

        # RNG initialization
        if hasattr(rng, 'random_integers'):
            self.rng = rng
        else:
            self.rng = numpy.random.RandomState(rng)

        print which_set
        print "Start is", start, "stop is", stop

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
	    #for i in indexes:
		#sequences.append(self.nodep[i])
	    #sequences = self.node[indexes]
            # sequences = self.samples_sequences[indexes]

            # Get random start point for ngram
            wis = [numpy.random.randint(0, len(s)-self.frame_length+1) for s in sequences]
            X = numpy.asarray([s[wi:self.frame_length+wi] for s, wi in zip(sequences, wis)])

            # Words mapped to integers greater than input max are set to 1 (unknown)
            X[X>=self.X_labels] = 1

            swaps = numpy.random.randint(0, self.frame_length - 1, len(X))
            y = numpy.zeros((len(X), self.frame_length - 1))
            y[numpy.arange(len(X)), swaps] = 1
            # print "Performing permutations...",
            for sample, swap in enumerate(swaps):
                X[sample, swap], X[sample, swap + 1] = \
                                                  X[sample, swap + 1], X[sample, swap]

            # Store the targets generated by these indices.
            self.lastY = (y, indexes)
            return X

        def getTarget(indexes):
            if numpy.array_equal(indexes, self.lastY[1]):
                return self.lastY[0]
            else:
                print "You can only ask for targets immediately after asking for those features"
                return None

        self.sourceFNs = {'features': getFeatures, 'targets': getTarget}

    def _parallel_load_data(self, start, stop, queue):
        #print "Starting to load data"
        #with tables.open_file(self.base_path) as f:
        #self.node = f.get_node(self.node_name)
        with tables.open_file(self.base_path) as f:
            if self.schwenk:
                table_name, index_name = '/phrases', '/indices'
                indices = f.get_node(index_name)[start:stop]
                first_word = indices[0]['pos']
                last_word = indices[-1]['pos'] + indices[-1]['length']
                words = f.get_node(table_name)[first_word:last_word]
                new_data = [words[i['pos']:i['pos']+i['length']] for i in indices]
            else:
                node = f.get_node(self.node_name)
                new_data = node[start:stop]
        queue.put(new_data)
        #print "Finished loading data"

    def _load_data(self, which_set, startstop):
        """
        Load the WMT14 data from disk.

        Parameters
        ----------
        which_set : str
            Subset of the dataset to use (either "train", "valid" or "test")
        """
        # TODO: Make files work with this terminology

        # Check which_set
        #if which_set not in ['train', 'valid', 'test']:
        #    raise ValueError(which_set + " is not a recognized value. " +
        #                     "Valid values are ['train', 'valid', 'test'].")
            
        # Load Data
        print startstop
        (start, stop) = startstop
	f = tables.open_file(self.base_path)

        if self.schwenk:
            table_name, index_name = '/phrases', '/indices'
            indices = f.get_node(index_name)[start:stop]
            first_word = indices[0]['pos']
            last_word = indices[-1]['pos'] + indices[-1]['length']
            words = f.get_node(table_name)[first_word:last_word]
            self.samples_sequences = [words[i['pos']:i['pos']+i['length']] for i in indices]
            self.num_examples = len(indices)
            f.close()
            return

        self.node = f.get_node(self.node_name)
        # with tables.open_file(self.base_path) as f:
        #     print "Loading n-grams..."
        #     node = f.get_node(self.node_name)
        if self._load_to_memory:
            if stop is not None:
                self.samples_sequences = self.node[start:stop]
            else:
                self.samples_sequences = self.node[start:]
            self.num_examples = len(self.samples_sequences)
            f.close()
        else:
            if stop is None:
                self.num_examples = self.node.nrows
            else:   
                self.num_examples = stop - start 
        print "Got", self.num_examples, "sentences"
        #self.samples_sequences = numpy.asarray(self.samples_sequences)
 
    def _validate_source(self, source):
        """
        Verify that all sources in the source tuple are provided by the
        dataset. Raise an error if some requested source is not available.

        Parameters
        ----------
        source : `tuple` of `str`
            Requested sources
        """
        for s in source:
            try:
                self.data_specs[1].index(s)
            except ValueError:
                raise ValueError("the requested source named '" + s + "' " +
                                 "is not provided by the dataset")

    

    def get_data_specs(self):
        """
        Returns the data_specs specifying how the data is internally stored.

        This is the format the data returned by `self.get_data()` will be.

        .. note::

            Once again, this is very hacky, as the data is not stored that way
            internally. However, the data that's returned by `.get()`
            _does_ respect those data specs.
        """
        return self.data_specs
    
    def _maybe_load_data(self):
       # print "In maybe load data"
        if self._num_since_last_load >= self._cache_delta and not self._loading:
            #print "need to load data"

            # If we would go over the end of the dataset by loading more data,
            # we start over from the beginning of the dataset.
            if (self._next_cache_index + self._cache_delta > 
                self._max_data_index):
                start = self._start
                stop = self._cache_delta + start
            else:
                start = self._next_cache_index
                stop = self._cache_delta + start

            self._next_cache_index = stop 
            self._loading = True
            p = Process(target=self._parallel_load_data, args=(start, stop, self._data_queue))
            p.start()

        if not self._data_queue.empty():
            #print "queue has stuff"
            self.samples_sequences = self.samples_sequences[self._cache_delta:] + self._data_queue.get()
            #print "Queue is empty", self._data_queue.empty()
            self._num_since_last_load = 0
            self._loading = False
            #print "got stuff from queue"

    def get(self, source, indexes):
        """
        .. todo::

            WRITEME
        """
        if self._using_cache:
            self._num_since_last_load += len(indexes)
            #print "new batch", self._num_examples_seen
            self._maybe_load_data()

        if type(indexes) is slice:
            indexes = numpy.arange(indexes.start, indexes.stop)
        self._validate_source(source)
        rval = []
        for so in source:
            batch = self.sourceFNs[so](indexes)
            rval.append(batch)
        return tuple(rval)

    def get_num_examples(self):
        return self.num_examples

    @functools.wraps(Dataset.iterator)
    def iterator(self, mode=None, batch_size=None, num_batches=None,
                 rng=None, data_specs=None, return_tuple=False):
        """
        .. todo::

            WRITEME
        """
        if data_specs is None:
            data_specs = self._iter_data_specs

        # If there is a view_converter, we have to use it to convert
        # the stored data for "features" into one that the iterator
        # can return.
        space, source = data_specs
        if isinstance(space, CompositeSpace):
            sub_spaces = space.components
            sub_sources = source
        else:
            sub_spaces = (space,)
            sub_sources = (source,)

        convert = []
        for sp, src in safe_zip(sub_spaces, sub_sources):
            convert.append(None)
 
        # TODO: Refactor
        print "In shuffle init!"
        if mode is None:
            if hasattr(self, '_iter_subset_class'):
                mode = self._iter_subset_class
            else:
                raise ValueError('iteration mode not provided and no default '
                                 'mode set for %s' % str(self))
        else:
            mode = resolve_iterator_class(mode)

        if batch_size is None:
            batch_size = getattr(self, '_iter_batch_size', None)
        #if num_batches is None:
        #    num_batches = getattr(self, '_iter_num_batches', None)
        num_batches = self._iter_num_batches 
	if rng is None and mode.stochastic:
            rng = self.rng
        return FiniteDatasetIterator(self,
                                     mode(self.num_examples, batch_size,
                                          num_batches, rng),
                                     data_specs=data_specs,
                                     return_tuple=return_tuple,
                                     convert=convert)

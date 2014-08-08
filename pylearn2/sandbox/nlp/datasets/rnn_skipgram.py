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
from pylearn2.utils import safe_zip, safe_izip
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
                 word_dict, char_dict,
                 start=0, stop=None, word_labels=None, char_labels=None,
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
        """
        super(H5RnnSkipgram, self).__init__(path, node, which_set, frame_length,
                 start=start, stop=stop, X_labels=word_labels,
		 _iter_num_batches=_iter_num_batches, rng=rng, 
                 load_to_memory=load_to_memory, cache_size=cache_size,
                 cache_delta=cache_delta, schwenk=schwenk)

        self._word_dict_path = word_dict
        self._char_dict_path = char_dict
        self._load_dicts()
        self.word_labels = word_labels
        features_space = SequenceDataSpace(IndexSpace(dim=1, max_labels=char_labels))
        features_source = 'features'

        targets_space = IndexSpace(dim=1, max_labels=self.word_labels)
        targets_source = 'targets'

        spaces = [features_space,  targets_space]
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
            # Get random source word index for "ngram"
            source_i = [numpy.random.randint(self.frame_length/2 +1, len(s)-self.frame_length/2, 1)[0] 
                        for s in sequences]
            target_i = [min(abs(int(numpy.random.normal(s_i, self.frame_length/3.0))), len(s)-1)
                        for s_i, s in safe_izip(source_i, sequences)]
            preX = [s[i] for i, s in safe_izip(source_i, sequences)]
            X = []
            def make_sequence(word):
                string = self._inv_words[word]
                #if len(string) < 1:
                    #print "Word index", word, "Returns empty word"
                seq = map(lambda c: [self._char_labels[c]], self._inv_words[word])
                #if len(seq) < 1:
                   # print "Word index", word, "Returns empty sequence", string
                seq.append([self._eow])
                return numpy.asarray(seq)

            for word in preX:
                X.append(make_sequence(word))
               # #####
               # min_len = 100
               # if len(X)<min_len:
               #     min_len=len(seq)
               # ####
            X = numpy.asarray(X)
            y = [numpy.asarray([s[i]]) for i, s in safe_izip(target_i, sequences)]
            y[y>=self.X_labels] = numpy.asarray([1])
            y = numpy.asarray(y)
            # Target Words mapped to integers greater than input max are set to 
            # 1 (unknown)

            # Store the targets generated by these indices.
            self.lastY = (y, indexes)
            return X

        def getTarget(indexes):
            if numpy.array_equal(indexes, self.lastY[1]):
                #y = numpy.transpose(self.lastY[0][:,source_index][numpy.newaxis])
                #print y
                #print y[-1]
                return self.lastY[0]
            else:
                print "You can only ask for targets immediately after asking for those features"
                return None
               
        # targetFNs = [
        #     lambda indexes: getTarget(0, indexes), lambda indexes: getTarget(1, indexes),
        #     lambda indexes: getTarget(2, indexes),lambda indexes: getTarget(3, indexes),
        #     lambda indexes: getTarget(4, indexes), 
        #     lambda indexes: getTarget(5, indexes)]
        # targetFNs = [(lambda indexes: getTarget(i, indexes)) for i in range(len(targets_space))]
        #self.sourceFNs = {'targets'+str(i): targetFNs[i] for i in range(len(targets_space))}
        self.sourceFNs['features'] =  getFeatures
        self.sourceFNs['targets'] = getTarget
      
    def _load_dicts(self):
        # word_dict_path = "/data/lisatmp3/pougetj/vocab.pkl"
        # char_dict_path = "/data/lisatmp3/pougetj/char_vocab.pkl"
        with open(self._word_dict_path) as f:
            word_labels = cPickle.load(f)
        self._inv_words = {v:k for k, v in word_labels.items()}
        with open(self._char_dict_path) as f:
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
        if self._iter_num_batches is not None:
            num_batches = self._iter_num_batches 
        subset_iterator = self._create_subset_iterator(
            mode=mode, batch_size=batch_size, num_batches=num_batches, rng=rng
        )
        return SequenceDatasetIterator(self, data_specs, subset_iterator,
                                       return_tuple=return_tuple)


    # @functools.wraps(Dataset.iterator)
    # def iterator(self, mode=None, batch_size=None, num_batches=None,
    #              rng=None, data_specs=None, return_tuple=False):
    #     """
    #     .. todo::

    #         WRITEME
    #     """
    #     if rng is None:
    #         rng = self.rng
    #     if mode is None or mode == 'shuffled_sequential':
    #         subset_iterator = ShuffledSequentialSubsetIterator(
    #             dataset_size=self.get_num_examples(),
    #             batch_size=batch_size,
    #             num_batches=num_batches,
    #             rng=rng,
    #             sequence_lengths=self._sequence_lengths
    #         )
    #     elif mode == 'sequential':
    #         subset_iterator = SequentialSubsetIterator(
    #             dataset_size=self.get_num_examples(),
    #             batch_size=batch_size,
    #             num_batches=num_batches,
    #             rng=None,
    #             sequence_lengths=self._sequence_lengths
    #         )
    #     else:
    #         raise ValueError('For sequential datasets only the '
    #                          'SequentialSubsetIterator and '
    #                          'ShuffledSequentialSubsetIterator have been '
    #                          'ported, so the mode `%s` is not supported.' %
    #                          (mode,))

    #     if data_specs is None:
    #         data_specs = self.data_specs
    #     return FiniteDatasetIterator(
    #         dataset=self,
    #         subset_iterator=subset_iterator,
    #         data_specs=data_specs,
    #         return_tuple=return_tuple
    #     )



        # if data_specs is None:
        #     data_specs = self._iter_data_specs

        # # TODO: Refactor
        # if mode is None or mode == 'shuffled_sequential':
        #     subset_iterator = ShuffledSequentialSubsetIterator(
        #         dataset_size=self.get_num_examples(),
        #         batch_size=batch_size,
        #         num_batches=num_batches,
        #         rng=rng,
        #         #sequence_lengths=self._sequence_lengths
        #     )
        # elif mode == 'sequential':
        #     subset_iterator = SequentialSubsetIterator(
        #         dataset_size=self.get_num_examples(),
        #         batch_size=batch_size,
        #         num_batches=num_batches,
        #         rng=None,
        #         #sequence_lengths=self._sequence_lengths
        #     )


        # # if mode is None:
        # #     if hasattr(self, '_iter_subset_class'):
        # #         # mode = self._iter_subset_class 
        # #     else:
        # #         raise ValueError('iteration mode not provided and no default '
        # #                          'mode set for %s' % str(self))
        # # else:
        # #     mode = resolve_iterator_class(mode)

        # if batch_size is None:
        #     batch_size = getattr(self, '_iter_batch_size', None)
        # #if num_batches is None:
        # #    num_batches = getattr(self, '_iter_num_batches', None)
        # num_batches = self._iter_num_batches 
        # if rng is None:# and mode.stochastic:
        #     rng = self.rng

        # if data_specs is None:
        #     data_specs = self.data_specs
        # return FiniteDatasetIterator(
        #     dataset=self,
        #     subset_iterator=subset_iterator,
        #     data_specs=data_specs,
        #     return_tuple=return_tuple
        # )


        # return FiniteDatasetIterator(self,
        #                              mode(self.num_examples, batch_size,
        #                                   num_batches, rng),
        #                              data_specs=data_specs,
        #                              return_tuple=return_tuple,
        #                              convert=convert)

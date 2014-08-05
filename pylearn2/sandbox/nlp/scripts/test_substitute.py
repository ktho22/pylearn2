import theano
import theano.tensor as T
import numpy as np
import os, sys, ipdb, tables
from pylearn2.sandbox.nlp.datasets.shuffle_substitution import H5Shuffle

path = '/data/lisatmp3/chokyun/mt/vocab.30k/bitexts.selected/binarized_text.en.h5'
node = 'none'
which_set = 'train'
frame_length = 7
start = 0
stop = 10
X_labels = 30000
schwenk = True

Dataobj = H5Shuffle(path,node,which_set,frame_length,start,stop,X_labels,load_to_memory=True,schwenk=schwenk)
Dataobj.sourceFNs['features']([0])

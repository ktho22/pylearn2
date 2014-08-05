import numpy as np
from collections import Counter
import tables, cPickle, ipdb
path = '/data/lisatmp3/chokyun/mt/vocab.30k/bitexts.selected/binarized_text.en.h5'
words_path =  '/data/lisatmp3/chokyun/mt/vocab.30k/bitexts.selected/vocab.en.pkl'

words = cPickle.load(open(words_path))
n_iter = 100000
f = tables.open_file(path)
table_name, index_name = '/phrases','/long_indices'

count = Counter()
nexam = len(f.get_node(index_name))
for i in xrange(n_iter):
    print '%d / %d' % (i,n_iter)
    indices = f.get_node(index_name)[int(i*nexam/float(n_iter)):int((i+1)*nexam/float(n_iter))]
    words = f.get_node(table_name)
    samples = [words[i['pos']:i['pos']+i['length']] for i in indices]
    for seq in samples:
        count += Counter(seq)
    ipdb.set_trace()


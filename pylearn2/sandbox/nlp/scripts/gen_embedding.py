import cPickle, sys, os
import numpy as np
from CharModel import CharModel

model_path = sys.argv[1]
chars_path = '/data/lisatmp3/devincol/data/translation_char_vocab.en.pkl'
vocab_path = '/data/lisatmp3/chokyun/mt/vocab.30k/bitexts.selected/vocab.en.pkl'
words_path = '/data/lisatmp3/devincol/data/translation_vocab_aschar.en.pkl'
savepath= '/data/lisatmp3/devincol/embeddings/'

print "Loading Data"
with open(vocab_path) as f:
    vocab = cPickle.load(f)
ivocab = {v:k for k,v in vocab.iteritems()}

with open(model_path) as f:
    pylearn2_model = cPickle.load(f)

with open(words_path) as f:
    words = cPickle.load(f)

with open(chars_path) as f:
    char_dict = cPickle.load(f)
inv_dict = {v:k for k,v in char_dict.items()}
inv_dict[0] = inv_dict[len(inv_dict.keys())-1]
unknown =  inv_dict[0]

print "Building Model"
import ipdb;ipdb.set_trace()
fpropNoProjLayer = pylearn2_model.layers[0].fprop
fpropProjLayer = lambda state_below: pylearn2_model.layers[1].fprop(pylearn2_model.layers[0].fprop(state_below))
model = CharModel(pylearn2_model, char_dict, fprop=fpropProjLayer, append_eow = True)
#model = CharModel(pylearn2_model, char_dict, fprop=fpropNoProjLayer)

x = model.genEmbeddings(ivocab)

# def closest(vec, n):
#     words = []
#     dists = [(cosine(vec,rnn_embeddings[i]), i) for i in range(len(rnn_embeddings))]
#     for k in range(n):
#         index = min(dists)[1]
#         dists[index] = (float("inf"),index)
#         words.append(index)
#     return words

 
savename = os.path.splitext(os.path.basename(model_path))[0]
save_at =os.path.join(savepath,savename)
np.save(save_at, x)

import ipdb;ipdb.set_trace()
import shutil
yamlname = os.path.splitext(model_path)[0] +'.yaml'
if os.path.exists(yamlname):
   shutil.copy2(yamlname,savepath)


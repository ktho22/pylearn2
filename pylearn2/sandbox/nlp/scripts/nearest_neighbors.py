import cPickle, sys, os
import numpy as np
from CharModel import CharModel

words_path = '/data/lisatmp3/devincol/data/translation_vocab_aschar.en.pkl'
model_path = sys.argv[1]

input_words=['cat','dog','France','france','Canada','Paris','pars','brother','mother','sister','dad','mom','pharmacy','farm','quite','quiet','quit','like','love','city','town','into','Committee','?','first','must','his','$','including','well']
with open(words_path) as f:
    words = cPickle.load(f)

with open(model_path) as f:
    model = cPickle.load(f)

model.words= words
map(model.displayStringRun,input_words)



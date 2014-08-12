import theano, cPickle, os, sys, ipdb
import numpy as np
import theano.tensor as T
from pylearn2.config import yaml_parse

chars_path = '/data/lisatmp3/devincol/data/translation_char_vocab.en.pkl'
vocab_path = '/data/lisatmp3/chokyun/mt/vocab.30k/bitexts.selected/vocab.en.pkl'
words_path = '/data/lisatmp3/devincol/data/translation_vocab_aschar.en.pkl'
savepath= '/data/lisatmp3/devincol/embeddings/'

def gen(savename):
    model = cPickle.load(open(savename))
    basename = os.path.splitext(os.path.basename(savename))[0]
    yamlname = os.path.splitext(savename)[0] +'.yaml'
    
    x = yaml_parse.load(open(yamlname)).dataset.X
    
    # If there is only projection layer ???
    #word_emb = model.layers[0].get_params()[0].get_value() 

    # If not
    space = model.get_input_space()
    inp = space.make_theano_batch()
    fprop = theano.function([inp], model.fprop(inp))

    
    
    save_at =os.path.join(savepath,basename) 
    np.save(save_at, word_emb)
    print save_at

    # f .yaml exists, move it!
    import shutil
    if os.path.exists(yamlname):
       shutil.copy2(yamlname,savepath)

if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print 'Processing ... ' + arg
        gen(arg)


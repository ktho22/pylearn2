import theano, cPickle, os, sys
import numpy as np
import theano.tensor as T
from pylearn2.sandbox.nlp.datasets.shuffle2 import H5Shuffle

savepath= '/data/lisatmp3/devincol/embeddings/'

def gen(savename):
    with open(savename) as f:
       model = cPickle.load(f)
       # The first layer is your projection layer, the embeddings are in the weight
       # matrix, which is returned by get_params(). This will give you a shared Theano
       # variable, which you can convert to a NumPy array using get_value()
       x = model.layers[0].get_params()[0].get_value() 
       savename = os.path.splitext(os.path.basename(savename))[0]
       np.save(os.path.join(savepath,savename), x)

if __name__ == "__main__":
    for arg in sys.argv:
        ext = os.path.splitext(arg)[1]
        if ext != '.pkl':
            continue
        print 'Processing ... ' + arg
        gen(arg)


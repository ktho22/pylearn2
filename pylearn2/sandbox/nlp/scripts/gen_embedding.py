import theano, cPickle, os, sys, ipdb
import numpy as np
import theano.tensor as T

savepath= '/data/lisatmp3/devincol/embeddings/'

def gen(savename):
    with open(savename) as f:
       ipdb.set_trace()
       model = cPickle.load(f)
       # The first layer is your projection layer, the embeddings are in the weight
       # matrix, which is returned by get_params(). This will give you a shared Theano
       # variable, which you can convert to a NumPy array using get_value()
       x = model.layers[0].get_params()[0].get_value() 
       savename = os.path.splitext(os.path.basename(savename))[0]
       save_at =os.path.join(savepath,savename) 
       np.save(save_at, x)
       print save_at

       # f .yaml exists, move it!
       import shutil
       yamlname = os.path.splitext(savename)[0] +'.yaml'
       if os.path.exists(yamlname):
           shutil.copy2(yamlname,savepath)

if __name__ == "__main__":
    for arg in sys.argv:
        ext = os.path.splitext(arg)[1]
        print 'Processing ... ' + arg
        gen(arg)


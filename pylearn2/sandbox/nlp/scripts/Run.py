import os, sys, ipdb, time
import pylearn2
from pylearn2.config import yaml_parse

dirname=os.path.abspath(os.path.dirname(__file__))

def get_hparams(fname,train):
   
    n_hids = '620'
    stop = 'None'
    
    save_path = os.path.join('result/%s_'%time.strftime("%m%d"))
    #if not os.path.exists(save_path):
    #    os.makedirs(save_path)
    save_path += fname +'_'+ n_hids 
        #+'_'+ stop

    if '.RecursiveConvolutionalLayer' in train:
        hparams = {
            'n_hids': eval(n_hids),
            'stop': eval(stop),
            'save_path': save_path}
    elif '.H5RnnSkipgram' in train:
        hparams = {
            'n_hids': eval(n_hids),
            'stop': eval(stop),
            'save_path': save_path}
    elif 'Shuffle' in fname:
        hparams = {
            'n_hids': eval(n_hids),
            'stop': eval(stop),
            'save_path': save_path}
    elif 'Substitute' in fname:
        hparams = {
            'n_hids': eval(n_hids),
            'stop': eval(stop),
            'save_path': save_path}
    elif 'partial_bag' in fname:
        hparams = {
            'n_hids': eval(n_hids),
            'stop': eval(stop),
            'save_path': save_path}
    else:
        raise ValueError()
    return hparams

for arg in sys.argv:
    fname,ext = os.path.splitext(arg)
    if ext != '.yaml':
        continue
    with open(os.path.join(dirname,arg),'r') as f:
        train= f.read()
    hparams = get_hparams(fname,train)
    savename=hparams['save_path']+'.yaml'
    hparams['save_path']+='.pkl'
    
    train = train % (hparams)
    print train
    
    fp = open(savename,'w')
    fp.write(train)
    fp.close()
    
    train_loop = yaml_parse.load(train)
    train_loop.main_loop()
    
    print hparams['save_path']


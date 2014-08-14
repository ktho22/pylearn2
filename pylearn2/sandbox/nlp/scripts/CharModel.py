import numpy as np
import theano as t
from scipy.spatial.distance import cosine

class CharModel():
   def __init__(self, model, char_dict, embeddings=None, fprop=None, words=None, append_eow=False):
      self.append_eow = append_eow
      space = model.get_input_space()
      data, mask = space.make_theano_batch(batch_size=1)
      self.fprop = t.function([data, mask], fprop((data,mask)))
      # if isinstance(batch, tuple):
      #    x = [b for b in batch]
      # else:
      #    x = [batch]
      # if fprop is not None:
      #    self.fprop = t.function([batch[0], batch[1]], fprop(batch))
      # else:
      #    self.fprop = t.function(x, model.fprop(batch))
      self.words = words
      self.embeddings = embeddings
      self.char_dict = char_dict
      self.ichar_dict = {v:k for k,v in char_dict.iteritems()}

   def genEmbeddings(self, ivocab):
      self.embeddings = []
      for i in range(len(ivocab)):
         self.embeddings.append(self.runString(ivocab[i]))
      return self.embeddings

   def arrToString(self, arr):
      return reduce(lambda x,y: x+y, arr)
      
   def stringToArr(self,string):
      arr = [self.char_dict.get(c, 0) for c in string]
      return arr

   def closest(self, vec, n):
      assert (self.embeddings is not None), "You probably need to run genEmbeddings"
      words_ = []
      dists = [(cosine(vec, self.embeddings[i]), i) for i in range(30000)]
      for k in range(n):
         index = min(dists)[1]
         dists[index] = (float("inf"),index)
         words_.append(index)
      return words_
         
   def run_example(self, example):
      if self.append_eow:
         example.append(144)
      data = np.asarray([np.asarray([np.asarray([char])]) for char in example])
      mask = np.ones((data.shape[0], data.shape[1]), dtype='float32')    
      wordvec = self.fprop(data, mask)[0]
      return wordvec

   def findClose(self, wordvec): 
      indices = self.closest(wordvec, 15)
      close = [self.makeWord(i) for i in indices]
      return close
    
   def runString(self, string):
      return self.run_example(self.stringToArr(string))

   def displayStringRun(self,word):
      L = self.stringToArr(word)
      close = self.findClose(self.run_example(L))
      print word, ":", close

   def displayIndexRun(self, index):
      assert (self.words is not None), "You need to give words to the model"
      close = self.findClose(self.run_example(self.words[index]))
      print self.makeWord(index), ":", close
      
   def makeWord(self, i):
      w = np.asarray(map(lambda n: self.ichar_dict[n], self.words[i]))
      return self.arrToString(w)

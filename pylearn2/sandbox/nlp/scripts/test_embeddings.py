import cPickle, ipdb
import tables, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cosine
import theano as t

path = sys.argv[-1]
#path = '/data/lisa/exp/kimtaeho/char_embedding/0724_rconv_char_embeddings_800_1000.pkl'
embeddings_path = '/data/lisa/data/word2vec/embeddings.h5'
chars_path = '/data/lisa/data/word2vec/char_vocab.pkl'
_path = '/data/lisa/data/word2vec/characters.pkl'

print "Loading data"
with open(path) as f:
    model = cPickle.load(f)

with open(chars_path) as f:
    char_dict = cPickle.load(f)
inv_dict = {v:k for k,v in char_dict.items()}
inv_dict[0] = inv_dict[len(inv_dict.keys())-1]
print "unknown", inv_dict[0]

with tables.open_file('/data/lisa/data/word2vec/characters.h5') as f:
    node1 = f.get_node('/characters_valid')
    valid_chars = [char_sequence for char_sequence in node1]
    node2 = f.get_node('/characters_train')
    train_chars = [char_sequence for char_sequence in node2]
print len(valid_chars), "valid chars"
print len (train_chars), "train chars"

# all_chars = valid_chars + train_chars
all_chars = train_chars[:100]
print len(all_chars), "all chars"

with tables.open_file(embeddings_path) as f:
    node1 = f.get_node('/embeddings_valid')
    valid_embeddings = node1[:]
    node2 = f.get_node('/embeddings_train')
    train_embeddings = node2[:]
# all_embeddings = np.concatenate((valid_embeddings, train_embeddings))
all_embeddings = train_embeddings
print all_embeddings.shape
with open('/data/lisatmp3/devincol/normalization.pkl') as f:
    (means, stds) = cPickle.load(f)

print "Normalizing"
all_embeddings = (all_embeddings - means)/stds

# words = []

# print "Making words"
# for x in all_chars:
#     #print x
#     w = np.asarray(map(lambda n: inv_dict[n], x))
#     #print w
#     words.append(w)
# print "Have", len(words), "words"
# print "have", len(all_embeddings), "embeddings"

#print "Making KDTree"
#tree = cKDTree(all_embeddings)

space = model.get_input_space()
batch_var = space.make_theano_batch(batch_size=1)
ipdb.set_trace()
fprop = t.function([batch_var], model.layers[0].fprop(batch_var))
#fprop = t.function([batch_var], model.fprop(batch_var))
#print "Batch_var shape", batch_var.eval().shape

print "Calculating projections"


def arrToString(arr):
    return reduce(lambda x,y: x+y, arr.view('S2')[::2])

def makeWord(i):
    w = np.asarray(map(lambda n: inv_dict[n], all_chars[i]))
    return arrToString(w)

def stringToArr(string):
    return [char_dict[c] for c in string]

def closest(vec, n):
    words = []
    dists = [(cosine(vec,all_embeddings[i]), i) for i in range(100)]
    for k in range(n):
        index = min(dists)[1]
        dists[index] = (float("inf"),index)
        words.append(index)
    return words
    
def run_example(example):
    ipdb.set_trace()
    batch = np.asarray([np.asarray([np.asarray([char])]) for char in example])
    batch = space.np_format_as(batch, space)
    wordvec = fprop(batch)[0]
    return wordvec

def findClose(wordvec): 
    indices = closest(wordvec, 4)
    #(d, indices) = tree.query(wordvec, k=4)
    close = [makeWord(i) for i in indices]
    return close

def run_string(word):
    L = stringToArr(word)
    # Character embedding
    close = findClose(run_example(L))
    # Word embedding
    #close = findClose(L)
    print word, ":", close

def run_index(index):
    # Character embedding
    close = findClose(run_example(all_chars[index]))
    # Word embedding
    #close = findClose(all_chars[index])
    print makeWord(index), ":", close

if __name__ == "__main__":
    map(run_string, ['monarch', 'democracy', 'political', 'raspberry', 'blueberry', 'accomplishment', 'applying', 'application'])
    map(run_index, range(0, 10))

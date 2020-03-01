import numpy as np
import torch.nn as nn
import torch
import bcolz
import pickle

class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


vectors = bcolz.open('./GloVe/6B.50.dat')[:]
words = pickle.load(open('./GloVe/6B.50_words.pkl', 'rb'))
word_num = pickle.load(open('./GloVe/6B.50_idx.pkl', 'rb'))
# target_vocab = pickle.load(open('data/vocab.pkl','rb'))

glove = {i: vectors[word_num[i]] for i in words}


def weights(target_vocab):
    
    matrix_len = len(target_vocab)
    weight_matrix = np.zeros((matrix_len, 50))
    word_found = 0

    for i, word in enumerate(target_vocab.idx2word):
        
        try:     
            weight_matrix[i] = glove[word]
            word_found += 1
        
        except KeyError:     
            weight_matrix[i] = np.random.normal(scale=0.6, size=(50))
    
    return weight_matrix



def emb_layer(weight_matrix, non_train=False):
   
#     print('wm: ', weight_matrix)
    embed_num, embed_dim = weight_matrix.shape
    emb_layer = nn.Embedding(embed_num, embed_dim)
    Dict = {'weight': weight_matrix}
    emb_layer.weight.data.copy_(torch.from_numpy(Dict['weight']))
    
    if non_train:
        emb_layer.weight.requires_grad = False

    return emb_layer, embed_num, embed_dim

def embed(sentence, weight_matrix):
    # sentence b x 56
    ans = torch.zeros(sentence.size(0), sentence.size(1), 50)
    for b in range(sentence.size(0)):
        for i in range(sentence.size(1)):
            ans[b, i] = torch.FloatTensor(weight_matrix[sentence[b,i]])
            
    return ans





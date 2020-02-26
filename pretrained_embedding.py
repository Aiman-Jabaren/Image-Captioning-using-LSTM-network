import numpy as np
import torch.nn as nn
import torch
import bcolz
import pickle


vectors = bcolz.open('./GloVe/6B.50.dat')[:]
words = pickle.load(open('./GloVe/6B.50_words.pkl', 'rb'))
word_num = pickle.load(open('./GloVe/6B.50_idx.pkl', 'rb'))

glove = {i: vectors[word_num[i]] for i in words}


def weights(target_vocab):
    
    matrix_len = len(target_vocab)
    weight_matrix = np.zeros((matrix_len, 50))
    word_found = 0

    for i, word in enumerate(target_vocab):
        
        try:     
            weight_matrix[i] = glove[word]
            word_found += 1
        
        except KeyError:     
            weight_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim))
    
    return weight_matrix, word_found



def emb_layer(weight_matrix, non_train=False):
   
    embed_num, embed_dim = weight_matrix.shape
    emb_layer = nn.Embedding(embed_num, embed_dim)
    Dict = {'weight': weight_matrix}
    emb_layer.weight.data.copy_(torch.from_numpy(Dict['weight']))
    
    if non_train:
        emb_layer.weight.requires_grad = False

    return emb_layer, embed_num, embed_dim





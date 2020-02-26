import torch
from torch import nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence

class VanillaRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, n_layers):
        
        super(VanillaRNN, self).__init__()
        self.device = torch.device("cuda:0")
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        #self.init_weights()

    def forward(self, x, features, lengths):

        embeds = self.embedding(x)
        embeds = torch.cat((features.unsqueeze(1), embeds), 1)
        packed = pack_padded_sequence(embeds, lengths, batch_first=True)
        ops, hidden = self.rnn(packed)
        output = self.fc(ops[0])
        return output, hidden

    def init_weights(self):
        
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        #hidden states initialized to zeros for each sample


"""
model = VanillaRNN(in_size (vocab size), out_size (vocab_size), hidden_dim, n_layers, embedding_dim)
model.to(device)

criterion = nn.CrossEntropyLoss() #or NLLLoss with softmax layer in forward
optimizer = torch.optim.Adam()
"""

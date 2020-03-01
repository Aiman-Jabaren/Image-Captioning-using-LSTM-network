import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F

class RNNDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, depth, vocab_size, batch_size):
        super(Decoder, self).__init__()
        self.depth = depth
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda:0")
        
#        self.initial_fc = nn.Linear(encoded_feature_dim, embedding_dim)
        
        # Embed input vector into tensor
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.hidden_dim = hidden_dim
        # RNN layer will take the feature tensor and previous hidden layer as input

        self.rnn = nn.RNN(embedding_dim, hidden_dim, depth, batch_first=True)
        self.hidden = torch.zeros((depth, batch_size, hidden_dim))

        # Linear layer mapping hidden rep -> output vector
        # We will use the full vocab in output as well
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def resetHidden(self, batch_size):
        self.hidden = torch.zeros((self.depth, batch_size, self.hidden_dim)).to(self.device)
                       
    def forward(self, sentence, features, lengths):

        embeds = self.embedding(sentence)
        
        
        outputs = []
        rnnOut, self.hidden = self.rnn(features.unsqueeze(1), self.hidden)
        outputs.append(rnnOut)
        for i in range(embeds.shape[1] - 1):
            rnnOut, self.hidden = self.rnn(embeds[:,i,:].unsqueeze(1), self.hidden)
            outputs.append(rnnOut)

        fc_out = self.fc(torch.stack(outputs, 1).squeeze())

        _, indices = fc_out.max(2)

        return fc_out.permute([0,2,1])
    
    def generate_caption(self, features, maxSeqLen, temperature, stochastic=False):

        

        rnn_inp = features.unsqueeze(1)
        word_ids = []
        self.resetHidden(1)
        if stochastic:
            for i in range(maxSeqLen):
                rnn_out, _ = self.rnn(rnn_inp)
                fc_out = self.fc(rnn_out.squeeze(1))
                scores = F.softmax(fc_out, dim=1) / temperature
                indices = (torch.distributions.Categorical(scores)).sample()
                word_ids.append(indices)
                rnn_inp = self.embedding(indices).unsqueeze(1)
                
                
        else:
            for i in range(maxSeqLen):
                if i == 0:
                    rnn_out, hidden = self.rnn(rnn_inp)
                else:
                    rnn_out, hidden = self.rnn(rnn_inp, hidden)
                    

                fc_out = self.fc(rnn_out.squeeze(1))
                _, indices = fc_out.max(1)

                
                rnn_inp = self.embedding(indices).unsqueeze(1)

                word_ids.append(indices.cpu())
                
            
        word_ids = torch.stack(word_ids,1)

        
        return word_ids

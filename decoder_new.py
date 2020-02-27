import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(Decoder, self).__init__()
        
        self.device = torch.device("cuda:0")
        
        # Embed input vector into tensor
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.hidden_dim = hidden_dim
        # LSTM layer will take the feature tensor and previous hidden layer as input

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Linear layer mapping hidden rep -> output vector
        # We will use the full vocab in output as well
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sentence, features, lengths):
        print('sentence shape: ', sentence.size())
        embeds = self.embedding(sentence)
#         embeds = embeds.view()
        print('features size: ', features.size())
        print('embeds size: ', embeds.size())
        print('lengths: ', lengths)
        
        lstm_inp = torch.cat((features.unsqueeze(1), embeds), 1)
        print('lstm inp shape: ', lstm_inp.size())
        
        packed_sequence = pack_padded_sequence(lstm_inp, lengths, batch_first=True, enforce_sorted=False)
        
        lstm_out, _ = self.lstm(packed_sequence)
        
#         print('lstm out shape: ', lstm_out.size())

        temp = pad_packed_sequence(lstm_out, batch_first=True)
    
        print('temp shape: ', temp[0].size())
        
#         fc_out = self.fc(temp[0])
#         print('asdfsd ', len(lstm_out))
#         tempseq = []
#         for i in range(len(lstm_out)):
#             tempseq.append(lstm_out[i].float().to(self.device))
#             print('i : ', lstm_out[i].size())
# #         temp = pad_sequence([lstm_out[0],lstm_out[1],lstm_out[2],lstm_out[3]], batch_first=True)
#         temp = torch.stack(tempseq, dim=0)
#         print('temp shape: ', temp.size())
        fc_out = self.fc(temp[0])
        print('fc_out shape: ', fc_out.size())
        
#         hidden_outs, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        
#         vocab_space = self.fc(hidden_outs.view(len(sentence), -1))

        word_scores = F.softmax(fc_out, dim=2)
        return word_scores
    
    def generate_caption(self, features, maxSeqLen, temperature, stochastic=False):
        # TODO - function for generating caption without using teacher forcing (using network outputs)
        
        lstm_inp = features.unsqueeze(1)
        word_ids = []
        
        if stochastic:
            for i in range(maxSeqLen):
                lstm_out, _ = self.lstm(lstm_inp)
                fc_out = self.fc(lstm_out.squeeze(1))
                scores = F.softmax(fc_out, dim=1) / temperature
                indices = (torch.distributions.Categorical(scores)).sample()
                word_ids.append(indices)
                lstm_inp = self.embedding(indices).unsqueeze(1)
                
                
        else:
            for i in range(maxSeqLen):
                lstm_out, _ = self.lstm(lstm_inp)
                fc_out = self.fc(lstm_out.squeeze(1))
                _, indices = fc_out.max(1)
                word_ids.append(indices)
                lstm_inp = self.embedding(indices).unsqueeze(1)
            
        word_ids = torch.stack(word_ids,1)
        print('word ids shape: ', word_ids.size())
        print('word ids: ', word_ids)
        
        return word_ids

import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(Decoder, self).__init__()

        # Embed input vector into tensor
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.hidden_dim = hidden_dim
        # LSTM layer will take the feature tensor and previous hidden layer as input

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # Linear layer mapping hidden rep -> output vector
        # We will use the full vocab in output as well
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sentence, features, lengths):
        embeds = self.embedding(sentence)
        
        lstm_inp = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        packed_sequence = pack_padded_sequence(embeddings, lengths, batch_first=True)
        
        lstm_out, _ = self.lstm(packed_sequence)
        
        fc_out = self.fc(lstm_out[0])
        
#         hidden_outs, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        
#         vocab_space = self.fc(hidden_outs.view(len(sentence), -1))

        word_scores = F.softmax(fc_out, dim=1)
        return word_scores
    
    def generate_caption(self, features):
        # TODO - function for generating caption without using teacher forcing (using network outputs)
import torch.nn as nn

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

    def forward(self, sentence):
        embeds = self.embedding(sentence)
        
        hidden_outs, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        
        vocab_space = self.fc(hidden_outs.view(len(sentence), -1))

        word_scores = F.log_softmax(vocab_space, dim=1)
        return word_scores

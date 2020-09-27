import torch
import torchvision


class Encoder(torch.nn.Module):

    def __init__(self, embedding_dim):
        super(Encoder, self).__init__()

        self.model = torchvision.models.vgg19(pretrained=True)
        self.last = torch.nn.Linear(
            self.model.classifier[-1].in_features, embedding_dim)
        self.model.classifier[-1] = self.last

        self.last.weight.data.normal_(0.0, 0.02)
        self.last.bias.data.fill_(0)

    def forward(self, x):
        return self.model(x)


class Decoder(torch.nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(Decoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.lstm = torch.nn.LSTM(embedding_dim, hidden_dim)
        self.last = torch.nn.Linear(hidden_dim, vocab_size)

        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.last.weight.data.uniform_(-0.1, 0.1)
        self.last.bias.data.fill_(0)

    def forward(self, features, caption):
        seq_length = len(caption) + 1
        embeds = self.embedding(caption)
        embeds = torch.cat((features, embeds), 0)
        lstm_out, _ = self.lstm(embeds.unsqueeze(1))
        return self.last(lstm_out.view(seq_length, -1))

    def greedy(self, cnn_out, seq_len=20):
        ip = cnn_out
        hidden = None
        ids_list = []
        for t in range(seq_len):
            lstm_out, hidden = self.lstm(ip.unsqueeze(1), hidden)
            # generating single word at a time
            linear_out = self.last(lstm_out.squeeze(1))
            word_caption = linear_out.max(dim=1)[1]
            ids_list.append(word_caption)
            ip = self.embedding(word_caption)
        return ids_list

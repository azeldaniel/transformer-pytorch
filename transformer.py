import torch
import math


class Transformer(torch.nn.Module):

    def init(self, num_tokens, num_inputs, num_heads, num_hidden, num_layers, dropout=0.5):
        super(Transformer, self).__init__()

        self.input_mask = None
        self.num_inputs = num_inputs

        self.position_encoder = PositionEncoder(num_inputs, dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                num_inputs, num_heads, num_hidden, dropout),
            num_layers
        )

        self.encoder = torch.nn.Embedding(num_tokens, num_inputs)
        self.decoder = torch.nn.Linear(num_inputs, num_tokens)

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        self.encoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()

    def _generate_square_subnet_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float(
            '-inf')).masked_fill(mask == 1, float(0.0))

    def forward(self, input):
        if(self.input_mask is None or self.input_mask.size(0) != input.size(0)):
            device = input.device
            mask = self._generate_square_subnet_mask(input.size(0)).to(device)
            self.input_mask = mask

        input = self.encoder(input)*math.sqrt(self.num_inputs)
        input = self.position_encoder(input)
        output = self.transformer_encoder(input, self.input_mask)
        return self.decoder(output)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

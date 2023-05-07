import torch
import torch.nn as nn


class NTGEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(NTGEncoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, X, state):
        X = self.embedding(X)

        output, state = self.rnn(X, state)
        hid0 = self.fc(torch.cat((state[0, :, :], state[1, :, :]), dim=1)).unsqueeze(0)
        hid1 = self.fc(torch.cat((state[-2, :, :], state[-1, :, :]), dim=1)).unsqueeze(0)
        state = torch.cat((hid0, hid1), dim=0)
        return output, state

    def init_state(self, batch_size):
        state = torch.zeros(2 * self.num_layers, batch_size, self.hidden_size, device=torch.device('cuda'))
        return state


class NTGDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(NTGDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size + hidden_size, hidden_size, num_layers, batch_first=False)
        self.dense = nn.Linear(hidden_size, vocab_size)

    def init_state(self, enc_outputs):
        return enc_outputs[1]

    def forward(self, X, state):
        # print(X.shape)
        # print(state.shape)
        X = self.embedding(X).permute(1, 0, 2)
        context = state[-1].repeat(X.shape[0], 1, 1)
        X_and_context = torch.cat((X, context), 2)

        output, state = self.rnn(X_and_context, state)
        output = self.dense(output).permute(1, 0, 2)
        # output[:, :, -1] = torch.tanh(output[:, :, -1]) * 0.5 + 0.5

        return output, state


class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X):
        enc_state = self.encoder.init_state(batch_size=enc_X.shape[0])
        enc_X = enc_X.permute(1, 0, 2)
        # enc_outputs = 0
        for i in range(enc_X.shape[0]):
            if (i == 0):
                enc_outputs = self.encoder(enc_X[i], enc_state)
            else:
                enc_outputs = enc_outputs + self.encoder(enc_X[i], enc_state)
        dec_state = self.decoder.init_state(enc_outputs)
        return self.decoder(dec_X, dec_state)

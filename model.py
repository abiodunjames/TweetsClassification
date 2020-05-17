import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):
    """
    This is a simple RNN architecture
    """

    def __init__(self, input_size, hidden_size, embed_dim, output_size):
        """
        Initialize the model by settingg up the various layers.
        """
        super(RNN, self).__init__()

        self.embed = nn.Embedding(input_size, embed_dim)
        self.rnn = nn.RNN(embed_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, tweet):
        """
        Perform a forward pass of our model on input.
        """
        # tweet = [len, batch_size]
        embedded = self.embed(tweet)
        output, hidden = self.rnn(embedded)
        output = self.fc(hidden.squeeze(0))
        return output


class LSTMClassifier(nn.Module):
    """
    This an LSTM architecture
    """

    def __init__(
        self,
        input_dim,
        embed_dim,
        hidden_dim,
        output_dim,
        n_layers,
        bidirectional,
        dropout,
        pad_idx,
        batch_first,
    ):
        """
        Initialize the model by settingg up the various layers.
        """
        super().__init__()

        self.embedding = nn.Embedding(input_dim, embed_dim, padding_idx=pad_idx)
        self.batch_first = batch_first
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=n_layers,
            bidirectional=bidirectional,
            dropout=dropout,
            batch_first=self.batch_first,
        )
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()

    def forward(self, tweet, length):
        """
        Perform a forward pass of our model on input.
        """
        embeds = self.dropout(self.embedding(tweet))
        # embeds = [batch_size, length, embed dim]
        packed_embeds = pack_padded_sequence(
            embeds, length, batch_first=self.batch_first, enforce_sorted=False
        )

        lstm_out, (hidden, _) = self.lstm(packed_embeds)
        output, output_lengths = pad_packed_sequence(
            lstm_out, batch_first=self.batch_first
        )
        # output = [batch_size, length, hidden_dim * num]
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        result = self.fc(hidden)

        return self.sig(result)

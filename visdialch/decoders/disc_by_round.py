import torch
from torch import nn
from visdialch.utils import DynamicRNN

class Disc_by_round_Decoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config
        self.nhid = config["lstm_hidden_size"]

        self.word_embed = nn.Embedding(
            len(vocabulary),
            config["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.option_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )
        self.a2a = nn.Linear(self.nhid * 2, self.nhid) # this is useless in this version
        # Options are variable length padded sequences, use DynamicRNN.
        self.option_rnn = DynamicRNN(self.option_rnn)

    def forward(self, encoder_output, batch):
        """Given `encoder_output` + candidate option sequences, predict a score
        for each option sequence.

        Parameters
        ----------
        encoder_output: torch.Tensor
            Output from the encoder through its forward pass.
            (batch_size, num_rounds, lstm_hidden_size)
        """
        options = batch["opt"]
        batch_size, num_options, max_sequence_length = options.size()
        options = options.contiguous().view(-1, max_sequence_length)

        options_length = batch["opt_len"]
        options_length = options_length.contiguous().view(-1)

        options_embed = self.word_embed(options)  # b*100 20 300
        _, (options_feat, _) = self.option_rnn(options_embed, options_length)  # b*100 512
        options_feat = options_feat.view(batch_size, num_options, self.nhid)


        encoder_output = encoder_output.unsqueeze(1).repeat(1, num_options, 1)
        scores = torch.sum(options_feat * encoder_output, -1)
        scores = scores.view(batch_size, num_options)

        return scores

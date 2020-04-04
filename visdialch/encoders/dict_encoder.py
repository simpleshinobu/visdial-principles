
import torch
from torch import nn
import numpy as np
from visdialch.utils import DynamicRNN

class Dict_Encoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config
        self.dropout = config['dropout']
        self.nhid = config['lstm_hidden_size']
        self.img_feature_size = config['img_feature_size']
        self.ninp = config['word_embedding_size']

        self.word_embed = nn.Embedding(
            len(vocabulary),
            config["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX,
        )
        self.hist_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )
        self.option_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )

        self.dropout = nn.Dropout(p=config["dropout_fc"])
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.option_rnn = DynamicRNN(self.option_rnn)
        self.Wc = nn.Linear(self.nhid * 2, self.nhid)
        self.Wd = nn.Linear(self.nhid, self.nhid)
        self.Wall = nn.Linear(self.nhid, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
        initial_path = 'data/100ans_feature.npy'
        initial_answer_feat = np.load(initial_path)
        self.user_dict = nn.Parameter(torch.FloatTensor(initial_answer_feat))

    def through_dict(self, output_feat): #b x 100 x 512 -> b 100 x512
        batch_size = output_feat.size(0)
        q_size = output_feat.size(1)
        dict_size = self.user_dict.size(0)
        dict_feat = self.user_dict

        q_emb = output_feat.view(batch_size * q_size, -1, self.nhid)
        d_emb = self.Wd(dict_feat).view(-1, dict_size, self.nhid)
        all_score = self.Wall(
            self.dropout(
                torch.tanh(d_emb.repeat(batch_size * q_size, 1, 1) + q_emb.repeat(1, dict_size, 1))
            )
        ).view(batch_size * q_size, -1)
        dict_final_feat = torch.bmm(
            torch.softmax(all_score, dim = -1 )
                .view(batch_size* q_size,1,-1),dict_feat.view(-1, dict_size, self.nhid).repeat(batch_size* q_size, 1, 1))
        return dict_final_feat.view(batch_size,q_size,-1)

    def forward(self, batch):

        his = batch["hist"] # b rnd q_len*2
        batch_size, rnd, max_his_length = his.size()
        his = his.contiguous().view(-1, max_his_length)
        his_embed = self.word_embed(his)
        _, (his_feat, _) = self.hist_rnn(his_embed, batch["hist_len"].contiguous().view(-1)) # b*rnd step 1024
        his_feat = his_feat.view(batch_size, rnd, self.nhid)
        his_feat = torch.mean(his_feat, dim=1)
        his_feat = his_feat

        options = batch["opt"]
        batch_size, num_options, max_sequence_length = options.size()
        options = options.contiguous().view(-1, max_sequence_length)
        options_length = batch["opt_len"]
        options_length = options_length.contiguous().view(-1)
        options_embed = self.word_embed(options)
        _, (options_feat, _) = self.option_rnn(options_embed, options_length)
        options_feat = options_feat.view(batch_size, num_options, self.nhid)

        his_feat = his_feat.unsqueeze(1).repeat(1,options_feat.size(1),1)
        cat_feat = torch.cat((his_feat,options_feat),dim=-1)
        cat_feat = self.dropout(self.Wc(cat_feat))

        output_feat = self.through_dict(cat_feat) # updated version (you also can try the key using (his; ques; ques and ans))
        scores = torch.sum(options_feat * output_feat, -1)
        scores = scores.view(batch_size, num_options)

        return scores #  out is b * 512

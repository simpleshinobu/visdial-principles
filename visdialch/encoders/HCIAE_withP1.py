import torch
from torch import nn
from visdialch.utils import DynamicRNN

class HCIAE_withP1_Encoder(nn.Module):
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
        self.ques_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )
        self.dropout = nn.Dropout(p=config["dropout_fc"])
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        ##q c att on img
        self.Wq2 = nn.Linear(self.nhid, self.nhid)
        self.Wh2 = nn.Linear(self.nhid, self.nhid)
        self.Wi2 = nn.Linear(self.img_feature_size, self.nhid)
        self.Wall2 = nn.Linear(self.nhid, 1)

        ##fusion
        self.Wq3 = nn.Linear(self.nhid , self.nhid )
        self.Wc3 = nn.Linear(self.nhid , self.nhid )
        self.fusion = nn.Linear(self.nhid + self.img_feature_size, self.nhid)
        ###cap att img
        self.Wc4 = nn.Linear(self.nhid , self.nhid)
        self.Wi4 = nn.Linear(self.img_feature_size, self.nhid)
        self.Wall4 = nn.Linear(self.nhid, 1)
        self.q_multi1 = nn.Linear(self.nhid, self.nhid)
        self.q_multi2 = nn.Linear(self.nhid, 3)

        ##q att on h
        self.Wq1 = nn.Linear(self.nhid, self.nhid)
        self.Wh1 = nn.Linear(self.nhid, self.nhid)
        self.Wqh1 = nn.Linear(self.nhid, 1)

        ###his on q
        self.Wcs5 = nn.Sequential(self.dropout,nn.Linear(self.nhid , self.nhid ))
        self.Wq5 = nn.Sequential(self.dropout,nn.Linear(self.nhid , self.nhid ))
        self.Wall5 = nn.Linear(self.nhid, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    def qh_att_on_img(self, ques_feat, his_feat, img_feat):
        batch_size = ques_feat.size(0)
        region_size = img_feat.size(1)
        device = ques_feat.device
        q_emb = self.Wq2(ques_feat).view(batch_size, -1, self.nhid)
        i_emb = self.Wi2(img_feat).view(batch_size, -1, self.nhid)
        h_emb = self.Wh2(his_feat).view(batch_size, -1, self.nhid)
        all_score = self.Wall2(
            self.dropout(
                torch.tanh(i_emb + q_emb.repeat(1, region_size, 1)+ h_emb.repeat(1, region_size, 1))
            )
        ).view(batch_size, -1)
        img_final_feat = torch.bmm(
            torch.softmax(all_score, dim = -1 )
                .view(batch_size,1,-1),img_feat)
        return img_final_feat.view(batch_size,-1)
    def ques_att_on_his(self, ques_feat, his_feat):
        batch_size = ques_feat.size(0)
        rnd = his_feat.size(1)
        device = ques_feat.device
        q_emb = self.Wq1(ques_feat).view(batch_size, -1, self.nhid)
        h_emb = self.Wh1(his_feat)
        score = self.Wqh1(
            self.dropout(
                torch.tanh(h_emb + q_emb.repeat(1, rnd, 1))
            )
        ).view(batch_size,-1)
        weight = torch.softmax(score, dim = -1 )
        atted_his_feat = torch.bmm(weight.view(batch_size,1,-1) ,his_feat)
        return atted_his_feat
    #####################################################

    def forward(self, batch):
        img = batch["img_feat"] # b 36 2048
        ques = batch["ques"] # b q_len
        his = batch["hist"] # b rnd q_len*2
        batch_size, rnd, max_his_length = his.size()
        ques_len = batch["ques_len"]

        # embed questions
        ques_location = batch['ques_len'].view(-1).cpu().numpy() - 1
        ques_embed = self.word_embed(ques) # b 20 300
        q_output, _ = self.ques_rnn(ques_embed, ques_len.view(-1)) # b rnd 1024
        ques_feat = q_output[range(batch_size), ques_location,:]

        ####his emb
        his = his.contiguous().view(-1, max_his_length)
        his_embed = self.word_embed(his) # b*rnd 40 300
        _, (his_feat, _) = self.hist_rnn(his_embed, batch["hist_len"].contiguous().view(-1)) # b*rnd step 1024
        his_feat = his_feat.view(batch_size, rnd, self.nhid)
        q_att_img_feat = self.qh_att_on_img(ques_feat, his_feat[:,0], img).view(batch_size, -1)

        fused_vector = torch.cat((ques_feat, q_att_img_feat), dim = -1)
        fused_vector = self.dropout(fused_vector)
        fused_embedding = torch.tanh(self.fusion(fused_vector)).view(batch_size, -1)

        return fused_embedding #  out is b * 512

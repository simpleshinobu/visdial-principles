import torch
from torch import nn

from visdialch.utils import DynamicRNN


class LF_Enhanced_Encoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config
        self.dropout = config['dropout']
        self.nhid = config['lstm_hidden_size']
        self.img_feature_size = config['img_feature_size']
        self.ninp = config['word_embedding_size']
        self.head_num = config['head_num']
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
        self.cap_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
        )
        self.dropout = nn.Dropout(p=config["dropout_fc"])
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)
        self.cap_rnn = DynamicRNN(self.cap_rnn)

        ##q c att on img
        self.Wq2 = nn.Sequential(self.dropout, nn.Linear(self.nhid * 2, self.nhid))
        self.Wi2 = nn.Sequential(self.dropout, nn.Linear(self.img_feature_size, self.nhid))
        self.Wall2 = nn.Linear(self.nhid, 1)

        # q_att_on_cap
        self.Wqs3 = nn.Sequential(self.dropout, nn.Linear(self.nhid, self.nhid))
        self.Wcs3 = nn.Sequential(self.dropout, nn.Linear(self.nhid, self.nhid))
        self.Wc3 = nn.Sequential(self.dropout, nn.Linear(self.nhid, self.nhid))
        self.Wall3 = nn.Linear(self.nhid, 1)
        self.c2c = nn.Sequential(self.dropout, nn.Linear(self.ninp, self.nhid))

        # c_att_on_ques
        self.Wqs5 = nn.Sequential(self.dropout, nn.Linear(self.nhid, self.nhid))
        self.Wcs5 = nn.Sequential(self.dropout, nn.Linear(self.nhid, self.nhid))
        self.Wq5 = nn.Sequential(self.dropout, nn.Linear(self.nhid, self.nhid))
        self.Wall5 = nn.Linear(self.nhid, 1)
        self.q2q = nn.Sequential(self.dropout, nn.Linear(self.ninp, self.nhid))
        # q att on h
        self.Wq1 = nn.Sequential(self.dropout, nn.Linear(self.nhid, self.nhid))
        self.Wh1 = nn.Sequential(self.dropout, nn.Linear(self.nhid, self.nhid))
        self.Wqh1 = nn.Linear(self.nhid, 1)
        ###cap att img
        self.Wc4 = nn.Sequential(self.dropout, nn.Linear(self.nhid * 2, self.nhid))
        self.Wi4 = nn.Sequential(self.dropout, nn.Linear(self.img_feature_size, self.nhid))
        self.Wall4 = nn.Linear(self.nhid, 1)
        ##fusion
        self.fusion_1 = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(self.nhid * 2 + self.img_feature_size + self.nhid, self.nhid),
            nn.LeakyReLU()
        )
        self.fusion_2 = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(self.nhid * 2 + self.img_feature_size + self.nhid, self.nhid),
            nn.LeakyReLU()
        )
        self.fusion_3 = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(self.nhid * 2 + self.img_feature_size + self.nhid, self.nhid),
            nn.LeakyReLU()
        )
        self.q_ref = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(self.nhid * 2, self.nhid),
            nn.LeakyReLU(),
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(self.nhid, 2),
            nn.LeakyReLU()
        )
        self.q_multi = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(self.nhid * 2, self.nhid),
            nn.LeakyReLU(),
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(self.nhid, 3),
            nn.LeakyReLU()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def q_att_on_cap(self, ques_prin, cap_prin, cap_feat, cap_len, cap_emb):
        batch_size = cap_feat.size(0)
        capfeat_len = cap_feat.size(1)
        q_emb = self.Wqs3(ques_prin).view(batch_size, -1, self.nhid)
        c_emb = self.Wcs3(cap_prin).view(batch_size, -1, self.nhid)
        cap_feat_new = self.Wc3(cap_feat)
        cap_score = self.Wall3(
            self.dropout(
                torch.tanh(cap_feat_new + q_emb.repeat(1, capfeat_len, 1) + c_emb.repeat(1, capfeat_len, 1))
            )
        ).view(batch_size, -1)
        mask = cap_score.detach().eq(0)
        for i in range(batch_size):
            mask[i, cap_len[i]:] = 1
        cap_score.masked_fill_(mask, -1e5)
        weight = torch.softmax(cap_score, dim=-1)
        final_cap_feat = torch.bmm(
            weight.view(batch_size, 1, -1),
            cap_emb).view(batch_size, -1)
        final_cap_feat = self.c2c(final_cap_feat)
        return final_cap_feat

    def c_att_on_ques(self, cap_prin, ques_prin, ques_feat, ques_len, ques_emb):
        batch_size = ques_feat.size(0)
        quesfeat_len = ques_feat.size(1)
        q_emb = self.Wqs5(ques_prin).view(batch_size, -1, self.nhid)
        c_emb = self.Wcs5(cap_prin).view(batch_size, -1, self.nhid)
        ques_feat_new = self.Wq5(ques_feat)
        ques_score = self.Wall5(
            self.dropout(
                torch.tanh(ques_feat_new + q_emb.repeat(1, quesfeat_len, 1) + c_emb.repeat(1, quesfeat_len, 1))
            )
        ).view(batch_size, -1)
        mask = ques_score.detach().eq(0)
        for i in range(batch_size):
            mask[i, ques_len[i]:] = 1
        ques_score.masked_fill_(mask, -1e5)
        weight = torch.softmax(ques_score, dim=-1)
        final_ques_feat = torch.bmm(
            weight.view(batch_size, 1, -1),
            ques_emb).view(batch_size, -1)
        final_ques_feat = self.q2q(final_ques_feat)
        return final_ques_feat

    def q_att_on_img(self, ques_feat, img_feat):
        batch_size = ques_feat.size(0)
        region_size = img_feat.size(1)
        device = ques_feat.device
        q_emb = self.Wq2(ques_feat).view(batch_size, -1, self.nhid)
        i_emb = self.Wi2(img_feat).view(batch_size, -1, self.nhid)
        all_score = self.Wall2(
            self.dropout(
                torch.tanh(i_emb * q_emb.repeat(1, region_size, 1))
            )
        ).view(batch_size, -1)
        img_final_feat = torch.bmm(
            torch.softmax(all_score, dim=-1)
                .view(batch_size, 1, -1), img_feat)
        return img_final_feat.view(batch_size, -1)

    def c_att_on_img(self, cap_feat, img_feat):
        batch_size = cap_feat.size(0)
        region_size = img_feat.size(1)
        device = cap_feat.device
        c_emb = self.Wc4(cap_feat).view(batch_size, -1, self.nhid)
        i_emb = self.Wi4(img_feat).view(batch_size, -1, self.nhid)
        all_score = self.Wall4(
            self.dropout(
                torch.tanh(i_emb * c_emb.repeat(1, region_size, 1))
            )
        ).view(batch_size, -1)
        img_final_feat = torch.bmm(
            torch.softmax(all_score, dim=-1)
                .view(batch_size, 1, -1), img_feat)
        return img_final_feat.view(batch_size, -1)

    ################################################add h
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
        ).view(batch_size, -1)
        weight = torch.softmax(score, dim=-1)
        atted_his_feat = torch.bmm(weight.view(batch_size, 1, -1), his_feat)
        return atted_his_feat

    #####################################################

    def forward(self, batch):
        img = batch["img_feat"]  # b 36 2048
        ques = batch["ques"]  # b q_len
        his = batch["hist"]  # b rnd q_len*2
        batch_size, rnd, max_his_length = his.size()
        cap = his[:, 0, :]
        ques_len = batch["ques_len"]
        cap_len = batch["hist_len"][:, 0]

        # embed questions
        ques_location = batch['ques_len'].view(-1).cpu().numpy() - 1
        ques_embed = self.word_embed(ques)  # b 20 300
        q_output, _ = self.ques_rnn(ques_embed, ques_len.view(-1))  # b rnd 1024
        ques_encoded = q_output[range(batch_size), ques_location, :]

        # embed caption
        cap_location = cap_len.view(-1).cpu().numpy() - 1
        cap_emb = self.word_embed(cap.contiguous())
        c_output, _ = self.cap_rnn(cap_emb, cap_len.view(-1))
        cap_encoded = c_output[range(batch_size), cap_location, :]

        ####his emb
        his = his.contiguous().view(-1, max_his_length)
        his_embed = self.word_embed(his)  # b*rnd 40 300
        _, (his_feat, _) = self.hist_rnn(his_embed, batch["hist_len"].contiguous().view(-1))  # b*rnd step 1024
        his_feat = his_feat.view(batch_size, rnd, self.nhid)
        q_att_his_feat = self.ques_att_on_his(ques_encoded, his_feat).view(batch_size, self.nhid)  # b 512
        att_cap_feat_0 = self.q_att_on_cap(ques_encoded, cap_encoded, c_output, cap_len,
                                           cap_emb)  # shape: (batch_size, 2*nhid)
        att_ques_feat_0 = self.c_att_on_ques(cap_encoded, ques_encoded, q_output, ques_len, ques_embed)
        his_att_cap = self.q_att_on_cap(q_att_his_feat, cap_encoded, c_output, cap_len, cap_emb)

        att_ques_feat = torch.cat((ques_encoded, att_ques_feat_0), dim=-1)
        att_cap_feat = torch.cat((cap_encoded, att_cap_feat_0), dim=-1)
        # attended ques on img
        q_att_img_feat = self.q_att_on_img(att_ques_feat, img).view(batch_size, -1, self.img_feature_size)
        # attended cap on img
        c_att_img_feat = self.c_att_on_img(att_cap_feat, img).view(batch_size, -1, self.img_feature_size)
        cated_feature = torch.cat((q_att_img_feat, c_att_img_feat), dim=1)
        # refine image attention
        q_gs = torch.softmax((self.q_ref(torch.cat((att_ques_feat_0, his_att_cap), dim=-1))), dim=-1).view(batch_size,
                                                                                                           1, -1)
        final_img_feat = torch.bmm(q_gs, cated_feature).view(batch_size, -1)
        # final fusion
        fused_vector = torch.cat((att_ques_feat, final_img_feat, q_att_his_feat), dim=-1)
        fused_embedding_1 = self.fusion_1(fused_vector).view(batch_size, 1, -1)
        fused_embedding_2 = self.fusion_2(fused_vector).view(batch_size, 1, -1)
        fused_embedding_3 = self.fusion_3(fused_vector).view(batch_size, 1, -1)

        fused_embedding = torch.cat((fused_embedding_1, fused_embedding_2, fused_embedding_3), dim=1)
        q_multi = torch.softmax((self.q_multi(att_ques_feat)), dim=-1).view(batch_size, 1, -1)
        fuse_feat = torch.bmm(q_multi, fused_embedding).view(batch_size, -1)

        return fuse_feat  # out is b * 512

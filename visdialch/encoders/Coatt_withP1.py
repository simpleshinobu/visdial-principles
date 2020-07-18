import torch
from torch import nn
from visdialch.utils import DynamicRNN


class CoAtt_withP1_Encoder(nn.Module):
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
        ################################################################origin
        ##q c att on img
        self.Wq2 = nn.Linear(self.nhid, self.nhid)
        self.Wi2 = nn.Linear(self.img_feature_size, self.nhid)
        self.Wall2 = nn.Linear(self.nhid, 1)

        ###########################add
        ##q att on h
        self.Wq1 = nn.Linear(self.nhid, self.nhid)
        self.Wh1 = nn.Linear(self.nhid, self.nhid)
        self.Wi1 = nn.Linear(self.img_feature_size, self.nhid)
        self.Wqvh1 = nn.Linear(self.nhid, 1)
        ##hv att on q
        self.Wq3 = nn.Linear(self.nhid, self.nhid)
        self.Wh3 = nn.Linear(self.nhid, self.nhid)
        self.Wi3 = nn.Linear(self.img_feature_size, self.nhid)
        self.Wqvh3 = nn.Linear(self.nhid, 1)
        ## step4
        self.Wq4 = nn.Linear(self.nhid, self.nhid)
        self.Wh4 = nn.Linear(self.nhid, self.nhid)
        self.Wi4 = nn.Linear(self.img_feature_size, self.nhid)
        self.Wall4 = nn.Linear(self.nhid, 1)
        ##
        self.fusion = nn.Linear(self.nhid + self.img_feature_size, self.nhid)
        ########################################new
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
    ###ATT STEP1
    def q_att_on_img(self, ques_feat, img_feat):
        batch_size = ques_feat.size(0)
        region_size = img_feat.size(1)
        device = ques_feat.device
        q_emb = self.Wq2(ques_feat).view(batch_size, -1, self.nhid)
        i_emb = self.Wi2(img_feat).view(batch_size, -1, self.nhid)
        all_score = self.Wall2(
            self.dropout(
                torch.tanh(i_emb + q_emb.repeat(1, region_size, 1))
            )
        ).view(batch_size, -1)
        img_final_feat = torch.bmm(
            torch.softmax(all_score, dim = -1 )
                .view(batch_size,1,-1),img_feat)
        return img_final_feat.view(batch_size,-1)
    ###ATT STEP2
    def qv_att_on_his(self, ques_feat, img_feat, his_feat):
        batch_size = ques_feat.size(0)
        rnd = his_feat.size(1)
        device = ques_feat.device
        q_emb = self.Wq1(ques_feat).view(batch_size, -1, self.nhid)
        i_emb = self.Wi1(img_feat).view(batch_size, -1, self.nhid)
        h_emb = self.Wh1(his_feat)

        score = self.Wqvh1(
            self.dropout(
                torch.tanh(h_emb + q_emb.repeat(1, rnd, 1)+ i_emb.repeat(1, rnd, 1))
            )
        ).view(batch_size,-1)
        weight = torch.softmax(score, dim = -1 )
        atted_his_feat = torch.bmm(weight.view(batch_size,1,-1) ,his_feat)
        return atted_his_feat
    ###ATT STEP2
    def hv_att_in_ques(self, his_feat, img_feat, q_output, ques_len):
        batch_size = q_output.size(0)
        q_emb_length = q_output.size(1)
        device = his_feat.device
        q_emb = self.Wq3(q_output)
        i_emb = self.Wi3(img_feat).view(batch_size, -1, self.nhid)
        h_emb = self.Wh3(his_feat).view(batch_size, -1, self.nhid)
        score = self.Wqvh3(
            self.dropout(
                torch.tanh(q_emb + h_emb.repeat(1, q_emb_length, 1)+ i_emb.repeat(1, q_emb_length, 1))
            )
        ).view(batch_size,-1)
        mask = score.detach().eq(0)
        for i in range(batch_size):
            mask[i,ques_len[i]:] = 1
        score.masked_fill_(mask, -1e5)
        weight = torch.softmax(score, dim = -1 )
        atted_his_ques = torch.bmm(weight.view(batch_size,1,-1) , q_output)
        return atted_his_ques
    ###ATT STEP4
    def qh_att_in_img(self, ques_feat, his_feat, img_feat):
        batch_size = ques_feat.size(0)
        region_size = img_feat.size(1)
        q_emb = self.Wq4(ques_feat).view(batch_size, -1, self.nhid)
        h_emb = self.Wh4(his_feat).view(batch_size, -1, self.nhid)
        i_emb = self.Wi4(img_feat)
        all_score = self.Wall4(
            self.dropout(
                torch.tanh(i_emb + q_emb.repeat(1, region_size, 1) +h_emb.repeat(1, region_size, 1))
            )
        ).view(batch_size, -1)
        img_final_feat = torch.bmm(
            torch.softmax(all_score, dim = -1 )
                .view(batch_size,1,-1),img_feat)
        return img_final_feat.view(batch_size,-1), torch.softmax(all_score, dim = -1)

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

        ############### ATT step1: q att on img -> v_1
        img_atted_feat_v1 = self.q_att_on_img(ques_feat, img).view(batch_size, self.img_feature_size)
        ############### ATT step2: q v att on his -> h_f
        his_atted_feat_f = self.qv_att_on_his(ques_feat, img_atted_feat_v1, his_feat).view(batch_size, self.nhid)
        ############### ATT step3: v_1 h_f att on ques -> q_f
        ques_atted_feat_f = self.hv_att_in_ques(his_atted_feat_f, img_atted_feat_v1, q_output, ques_len).view(batch_size, self.nhid)
        ############### ATT step4: q_f h_f att on img -> v_f
        img_atted_feat_f, img_att = self.qh_att_in_img(ques_atted_feat_f, his_feat[:,0], img)
        img_atted_feat_f = img_atted_feat_f.view(batch_size, self.img_feature_size)

        fused_vector = torch.cat((img_atted_feat_f, ques_feat), dim = -1)
        fused_embedding = torch.tanh(self.fusion(fused_vector)).view(batch_size, -1)
        return fused_embedding #  out is b * 512

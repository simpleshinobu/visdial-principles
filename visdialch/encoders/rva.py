import torch
from torch import nn
from torch.nn import functional as F
from visdialch.utils import DynamicRNN


class RvaEncoder(nn.Module):
    def __init__(self, config, vocabulary):
        super().__init__()
        self.config = config

        self.word_embed = nn.Embedding(
            len(vocabulary),
            config["word_embedding_size"],
            padding_idx=vocabulary.PAD_INDEX
        )
        # if config["fix_word_embedding"] == True:
        #     self.word_embed.weight.requires_grad = False

        self.hist_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
            bidirectional=True
        )
        self.ques_rnn = nn.LSTM(
            config["word_embedding_size"],
            config["lstm_hidden_size"],
            config["lstm_num_layers"],
            batch_first=True,
            dropout=config["dropout"],
            bidirectional=True
        )
        # questions and history are right padded sequences of variable length
        # use the DynamicRNN utility module to handle them properly
        self.hist_rnn = DynamicRNN(self.hist_rnn)
        self.ques_rnn = DynamicRNN(self.ques_rnn)

        # img_feature_size = config["img_feature_size"] + config["img_loc_size"]
        img_feature_size = config["img_feature_size"]
        lstm_hidden_size = config["lstm_hidden_size"]
        word_embed_size = config["word_embedding_size"]
        self.img_feature_size = img_feature_size
        self.lstm_hidden_size = lstm_hidden_size
        self.word_embed_size = word_embed_size
        self.relu = nn.ReLU()

        # new: attention
        # embedding
        self.Wii = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(img_feature_size, lstm_hidden_size),
            nn.BatchNorm1d(lstm_hidden_size),
            self.relu
        )

        self.Wqi = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(word_embed_size, lstm_hidden_size),
            nn.BatchNorm1d(lstm_hidden_size),
            self.relu
        )

        self.Wq_fuse_g = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(word_embed_size, img_feature_size),
            nn.Sigmoid()
        )

        self.Wqq_ans = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.BatchNorm1d(lstm_hidden_size),
            self.relu
        )

        self.Wqq_ref = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.BatchNorm1d(lstm_hidden_size),
            self.relu
        )

        self.Wqq_inf = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(word_embed_size, lstm_hidden_size),
            nn.BatchNorm1d(lstm_hidden_size),
            self.relu
        )

        self.Whh_ref = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.BatchNorm1d(lstm_hidden_size),
            self.relu
        )

        self.Wqh_ref = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.BatchNorm1d(lstm_hidden_size),
            self.relu
        )

        # attention
        self.Wia = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(lstm_hidden_size, 1)
        )
        self.Wqa_ans = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(lstm_hidden_size, 1)
        )
        self.Wqa_ref = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(lstm_hidden_size, 1)
        )
        self.Wha_ans = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(lstm_hidden_size, 1)
        )
        self.Wha_ref = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(lstm_hidden_size, 1)
        )
        self.Wh_ref = nn.Linear(2, 1)

        # referring to history
        self.Wq_inf = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                lstm_hidden_size,
                2
            )
        )
        # fusion
        self.fusion_v = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                img_feature_size,
                lstm_hidden_size * 2
            ),
            nn.BatchNorm1d(lstm_hidden_size * 2),
            self.relu
        )
        self.fusion_q = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                lstm_hidden_size * 2,
                lstm_hidden_size * 2
            ),
            nn.BatchNorm1d(lstm_hidden_size * 2),
            self.relu
        )
        self.fusion = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                img_feature_size + lstm_hidden_size * 2 + lstm_hidden_size * 2,
                lstm_hidden_size * config["ans_cls_num"]
            )
        )
        self.fusion_cls = nn.Sequential(
            nn.Dropout(p=config["dropout_fc"]),
            nn.Linear(
                lstm_hidden_size * 2,
                config["ans_cls_num"]
            )
        )
        # other useful functions
        self.softmax = nn.Softmax(dim=-1)
        # self.G_softmax = F.gumbel_softmax()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, batch):
        img = batch["img_feat"]
        ques = batch["ques"]
        hist = batch["hist"]
        cap = hist[:, 0, :].unsqueeze(1)
        batch_size, num_rounds, _ = ques.size()
        hist_len_max = hist.size(-1)

        # embed questions
        ques_not_pad = (ques != 0).float()
        ques = ques.view(-1, ques.size(-1))
        ques_word_embed = self.word_embed(ques)
        ques_embed, _ = self.ques_rnn(ques_word_embed,
                                      batch['ques_len'])
        quen_len_max = ques_embed.size(1)
        loc = batch['ques_len'].view(-1).cpu().numpy() - 1
        ques_encoded_1 = ques_embed[range(num_rounds * batch_size), loc,
                         :self.lstm_hidden_size]
        ques_encoded_2 = ques_embed[:, 0, self.lstm_hidden_size:]
        ques_encoded = torch.cat((ques_encoded_1, ques_encoded_2), dim=-1)
        ques_encoded = ques_encoded.view(-1, num_rounds,
                                         ques_encoded.size(-1))
        ques_embed = ques_embed.view(-1, num_rounds, quen_len_max,
                                     ques_embed.size(-1))
        ques_word_embed = ques_word_embed.view(-1, num_rounds, quen_len_max, ques_word_embed.size(
            -1))

        # embed history
        hist = hist.view(-1, hist.size(-1))
        hist_word_embed = self.word_embed(hist)
        hist_embed, _ = self.hist_rnn(hist_word_embed, batch['hist_len'])
        loc = batch['hist_len'].view(-1).cpu().numpy() - 1
        hist_encoded_1 = hist_embed[range(num_rounds * batch_size), loc,
                         :self.lstm_hidden_size]
        hist_encoded_2 = hist_embed[:, 0, self.lstm_hidden_size:]
        hist_encoded = torch.cat((hist_encoded_1, hist_encoded_2), dim=-1)
        hist_encoded = hist_encoded.view(-1, num_rounds,
                                         hist_encoded.size(-1))

        # embed hist refering
        cap = hist.view(-1, num_rounds, hist.size(-1))[:, 0, :]
        cap_not_pad = (cap != 0).float().unsqueeze(1)
        hist_embed_ref, _ = self.ques_rnn(hist_word_embed,
                                          batch['hist_len'])
        hist_embed_ref = hist_embed_ref.view(-1, num_rounds, hist_len_max, hist_embed_ref.size(
            -1))
        cap_word_embed = hist_word_embed.view(-1, num_rounds, hist_len_max, hist_word_embed.size(-1))[:, 0, :,
                         :]
        cap_word_embed = cap_word_embed.unsqueeze(1)
        cap_embed = hist_embed_ref[:, :1, :, :]

        ques_ans_feat, _ = self.ques_ans_attention(ques_word_embed[:, -1].unsqueeze(1), ques_embed[:, -1].unsqueeze(1),
                                                   ques_not_pad[:, -1].unsqueeze(
                                                       1))
        ques_ref_feat, _ = self.ques_ref_attention(ques_word_embed, ques_embed,
                                                   ques_not_pad)
        cap_ref_feat, _ = self.ques_ref_attention(cap_word_embed, cap_embed,
                                                  cap_not_pad)
        his_feature, hist_logits = self.hist_ref_attention(hist_encoded, ques_encoded)

        _, img_att_ques = self.ATT_MODULE(img, ques_ref_feat)
        _, img_att_cap = self.ATT_MODULE(img, cap_ref_feat)

        ques_gs, ques_gs_prob = self.ques_inf(ques_ref_feat)
        hist_gs_set, _ = self.hist_inf(hist_logits)
        img_ans_feat, _ = self.img_hist_attention(
            img, img_att_ques, img_att_cap,
            ques_gs, hist_gs_set, ques_gs_prob,
        )
        img_ans_feat = self.fuse_img_ans(img_ans_feat, ques_ans_feat)

        fused_vector = torch.cat((img_ans_feat, ques_encoded, his_feature), -1)

        fused_embedding = torch.tanh(self.fusion(fused_vector))
        fused_embedding = fused_embedding.view(batch_size, num_rounds, -1, self.lstm_hidden_size)

        fused_att = self.softmax(self.fusion_cls(ques_encoded))
        fused_att = fused_att.view(batch_size, num_rounds, -1).unsqueeze(-1)

        fused_embedding = (fused_att * fused_embedding).sum(-2)
        return fused_embedding

    def ATT_MODULE(self, img, ques):
        batch_size = ques.size(0)
        num_rounds = ques.size(1)
        num_proposals = img.size(1)

        img_embed = img.view(-1, img.size(-1))
        img_embed = self.Wii(img_embed)
        img_embed = img_embed.view(batch_size, num_proposals, img_embed.size(-1))
        img_embed = img_embed.unsqueeze(1).repeat(1, num_rounds, 1,
                                                  1)

        ques_embed = ques.view(-1, ques.size(-1))
        ques_embed = self.Wqi(ques_embed)
        ques_embed = ques_embed.view(batch_size, num_rounds, ques_embed.size(-1))
        ques_embed = ques_embed.unsqueeze(2).repeat(1, 1, num_proposals,
                                                    1)

        # point-wise
        att_embed = F.normalize(ques_embed * img_embed, p=2,
                                dim=-1)
        att_embed = self.Wia(att_embed).squeeze(-1)
        att = self.softmax(att_embed)
        img_att = torch.sum(att.unsqueeze(-1) * img.unsqueeze(1), dim=-2)

        return img_att, att

    def ques_ans_attention(self, ques_word, ques_embed, ques_not_pad):
        batch_size = ques_word.size(0)
        num_rounds = ques_word.size(1)
        quen_len_max = ques_word.size(2)

        ques_embed = ques_embed.contiguous().view(-1, ques_embed.size(-1))
        ques_embed = self.Wqq_ans(ques_embed)
        ques_embed = ques_embed.contiguous().view(batch_size, num_rounds, quen_len_max, ques_embed.size(-1))
        ques_norm = F.normalize(ques_embed, p=2, dim=-1)

        att = self.Wqa_ans(ques_norm).squeeze(-1)
        att = self.softmax(att)
        att = att * ques_not_pad
        att = att / torch.sum(att, dim=-1, keepdim=True)
        ques_att = torch.sum(att.unsqueeze(-1) * ques_word, dim=-2)

        return ques_att, att

    def ques_ref_attention(self, ques_word, ques_embed, ques_not_pad):
        batch_size = ques_word.size(0)
        num_rounds = ques_word.size(1)
        quen_len_max = ques_word.size(2)

        ques_embed = ques_embed.contiguous().view(-1, ques_embed.size(-1))
        ques_embed = self.Wqq_ref(ques_embed)
        ques_embed = ques_embed.contiguous().view(batch_size, num_rounds, quen_len_max, ques_embed.size(-1))
        ques_norm = F.normalize(ques_embed, p=2, dim=-1)

        att = self.Wqa_ref(ques_norm).squeeze(-1)
        att = self.softmax(att)
        att = att * ques_not_pad
        att = att / torch.sum(att, dim=-1, keepdim=True)
        ques_att = torch.sum(att.unsqueeze(-1) * ques_word, dim=-2)

        return ques_att, att

    def ques_inf(self, ques):
        batch_size = ques.size(0)
        num_rounds = ques.size(1)

        ques_embed = ques.contiguous().view(-1, ques.size(-1))
        ques_embed = self.Wqq_inf(ques_embed)
        ques_embed = ques_embed.contiguous().view(batch_size, num_rounds, ques_embed.size(-1))

        ques_logits = self.Wq_inf(ques_embed)
        ques_gs = F.gumbel_softmax(ques_logits.view(-1, 2), hard=True).view(-1, num_rounds, 2)
        ques_gs_prob = self.softmax(ques_logits)

        return ques_gs, ques_gs_prob

    def hist_inf(self, hist_logits):
        num_rounds = hist_logits.size(1)

        hist_gs_set = torch.zeros_like(hist_logits)
        hist_prob_set = torch.zeros_like(hist_logits)
        for i in range(num_rounds):
            hist_gs = F.gumbel_softmax(hist_logits[:, i, :(i + 1)], hard=True)
            hist_prob = self.softmax(hist_logits[:, i, :(i + 1)])
            hist_gs_set[:, i, :(i + 1)] = hist_gs
            hist_prob_set[:, i, :(i + 1)] = hist_prob

        return hist_gs_set, hist_prob_set

    def hist_ans_attention(self, hist, ques):
        batch_size = ques.size(0)
        num_rounds = ques.size(1)
        device = hist.device

        hist_embed = hist.contiguous().view(-1, hist.size(-1))
        hist_embed = self.Whh_ans(hist_embed)
        hist_embed = hist_embed.contiguous().view(batch_size, num_rounds, hist_embed.size(-1))
        hist_embed = hist_embed.unsqueeze(1).repeat(1, num_rounds, 1,
                                                    1)

        ques_embed = ques.contiguous().view(-1, ques.size(-1))
        ques_embed = self.Wqh_ans(ques_embed)
        ques_embed = ques_embed.contiguous().view(batch_size, num_rounds, ques_embed.size(-1))
        ques_embed = ques_embed.unsqueeze(2).repeat(1, 1, num_rounds,
                                                    1)

        att_embed = F.normalize(hist_embed * ques_embed, p=2,
                                dim=-1)
        att_embed = self.Wha_ans(att_embed).squeeze(-1)
        att = self.softmax(att_embed)
        att_not_pad = torch.tril(
            torch.ones(size=[num_rounds, num_rounds], requires_grad=False))
        att_not_pad = att_not_pad.to(device)
        att_masked = att * att_not_pad
        att_masked = att_masked / torch.sum(att_masked, dim=-1,
                                            keepdim=True)
        hist_att = torch.sum(att_masked.unsqueeze(-1) * hist.unsqueeze(1),
                             dim=-2)

        return hist_att, att_masked

    def hist_ref_attention(self, hist, ques):
        batch_size = ques.size(0)
        num_rounds = ques.size(1)
        device = hist.device

        hist_embed = hist.contiguous().view(-1, hist.size(-1))
        hist_embed = self.Whh_ref(hist_embed)
        hist_embed = hist_embed.contiguous().view(batch_size, num_rounds, hist_embed.size(-1))
        hist_embed = hist_embed.unsqueeze(1).repeat(1, num_rounds, 1,
                                                    1)

        ques_embed = ques.contiguous().view(-1, ques.size(-1))
        ques_embed = self.Wqh_ref(ques_embed)
        ques_embed = ques_embed.contiguous().view(batch_size, num_rounds, ques_embed.size(-1))
        ques_embed = ques_embed.unsqueeze(2).repeat(1, 1, num_rounds,
                                                    1)

        att_embed = torch.cat((hist_embed, ques_embed), dim=-1)
        att_embed = self.Wha_ref(att_embed).squeeze(-1)

        att = self.softmax(att_embed)
        att_not_pad = torch.tril(
            torch.ones(size=[num_rounds, num_rounds], requires_grad=False))
        att_not_pad = att_not_pad.to(device)
        att_masked = att * att_not_pad
        att_masked = att_masked / torch.sum(att_masked, dim=-1,
                                            keepdim=True)
        hist_att = torch.sum(att_masked.unsqueeze(-1) * hist.unsqueeze(1),
                             dim=-2)

        return hist_att, att_embed

    def img_hist_attention(self, img, img_att_ques, img_att_cap, ques_gs, hist_gs_set, ques_gs_prob):
        device = img.device
        num_rounds = ques_gs.size(1)
        num_proposals = img_att_ques.size(-1)
        batch_size = img.size(0)

        ques_prob_single = torch.Tensor(data=[1, 0]).view(1, -1).repeat(batch_size, 1)
        ques_prob_single = ques_prob_single.to(device)
        ques_prob_single.requires_grad = False

        img_att_refined = img_att_ques.data.clone().zero_()
        for i in range(num_rounds):
            if i == 0:
                img_att_temp = img_att_cap.view(-1, img_att_cap.size(-1))
            else:
                hist_gs = hist_gs_set[:, i, :(i + 1)]
                img_att_temp = torch.cat((img_att_cap, img_att_refined[:, :i, :]),
                                         dim=1)
                img_att_temp = torch.sum(hist_gs.unsqueeze(-1) * img_att_temp,
                                         dim=-2)
            img_att_cat = torch.cat((img_att_ques[:, i, :].unsqueeze(1), img_att_temp.unsqueeze(1)),
                                    dim=1)
            ques_prob_pair = ques_gs_prob[:, i, :]
            ques_prob = torch.cat((ques_prob_single, ques_prob_pair), dim=-1)
            ques_prob = ques_prob.view(-1, 2, 2)
            ques_prob_refine = torch.bmm(ques_gs[:, i, :].view(-1, 1, 2), ques_prob).view(-1, 1,
                                                                                          2)

            img_att_refined[:, i, :] = torch.bmm(ques_prob_refine, img_att_cat).view(-1,
                                                                                     num_proposals)
        img_att_feat = torch.bmm(img_att_refined, img)

        return img_att_feat, img_att_refined

    def fuse_img_ans(self, img, ques):

        batch_size = ques.size(0)
        num_rounds = ques.size(1)

        ques_embed = ques.contiguous().view(-1, ques.size(-1))
        ques_embed_g = self.Wq_fuse_g(ques_embed)
        ques_embed_g = ques_embed_g.contiguous().view(batch_size, num_rounds, ques_embed_g.size(-1))
        img_fused = img * ques_embed_g

        return img_fused
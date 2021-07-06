import torch
from torch import nn
import torch.nn.functional as F
from transformers import BertModel
from ivcml_util import show_type_tree


class DeepQNetIVCML(nn.Module):
    def __init__(self, d_fea, alpha, step_num):
        super(DeepQNetIVCML, self).__init__()
        self.d_fea = d_fea
        self.alpha = alpha
        self.step_num = step_num
        self.unit_fea_mlp_1 = nn.Linear(d_fea, d_fea)
        self.unit_fea_mlp_2 = nn.Linear(d_fea*2, d_fea)
        self.unit_fea_mlp_cls = nn.Linear(d_fea, 1)
        self.unit_fea_mlp_reg = nn.Linear(d_fea, 1)
        self.p_dropout = 0.9

        self.query_upd_mlp = nn.Linear(d_fea*3, d_fea)

        self.encode = "bert"
        self.freeze_bert = True
        if self.encode == "bert":
            self.bert_encoder = BertModel.from_pretrained("bert-base-uncased")
            self.text_encode_func = self.bert_encode
            for p in self.bert_encoder.parameters():
                p.requires_grad = not self.freeze_bert

    def forward(self, query_tok_ids, weight_observe, _fea_emb, _nei_mask, _move_gt=None):
        """
        :param query_tok_ids:
        :param weight_observe:
        :param _fea_emb: [B, N, D]
        :param _nei_mask:
        :param _move_gt:
        :return:
        """
        query_fea = self.bert_encode(query_tok_ids)
        move_pred, reward_pred = self.move(query_fea, weight_observe, _fea_emb, _nei_mask, _move_gt)
        return move_pred, reward_pred

    def move(self, query_fea, nei_w_obs, _fea_emb, _nei_mask, _move_gt):
        batch_size = query_fea.shape[0]
        nei_fea = torch.einsum('bsnv,bvd->bsnd', nei_w_obs, _fea_emb)
        seq_len = nei_w_obs.shape[1]

        nei_fea = self.unit_fea_mlp_1(nei_fea)
        nei_fea = F.relu(nei_fea)
        nei_fea = F.dropout(nei_fea, self.p_dropout)
        query_fea = query_fea.mean(dim=-2)  # mean of all words

        nei_move_pred_seq = []
        nei_reward_pred_seq = []
        for i_step in range(seq_len):
            nei_move_pred_step, nei_reward_pred_step = self.move_single_step(query_fea, nei_fea[:, i_step, :, :])
            move_idx_pred = torch.argmax(nei_move_pred_step, dim=-1, keepdim=False)
            nei_move_pred_seq.append(nei_move_pred_step)
            nei_reward_pred_seq.append(nei_move_pred_step)

            if _move_gt is None:
                move_idx = move_idx_pred
            else:
                # teacher forcing
                move_idx_gold = _move_gt[:, i_step]
                move_idx = move_idx_gold

            # Update query feature
            batch_dim = torch.arange(batch_size)
            pos_idx = (batch_dim, move_idx)
            pos_move_fea = nei_fea[:, i_step, :, :][pos_idx]

            # (sum of negative features - positive feature) / number of negative neighbors
            neg_move_fea = nei_fea[:, i_step, :, :].sum(dim=-2, keepdim=False) - pos_move_fea
            mask_wo_gt = _nei_mask[:, i_step, :]
            mask_wo_gt[pos_idx] = 0
            neg_move_num = mask_wo_gt.sum(dim=-1, keepdim=True)
            neg_move_num = neg_move_num.where(neg_move_num > 0, torch.ones_like(neg_move_num))
            neg_move_fea /= neg_move_num
            assert (neg_move_num > 0).all(), neg_move_num

            query_fea = self.update_query(query_fea, pos_move_fea, neg_move_fea)

        nei_move_pred_seq = torch.stack(nei_move_pred_seq, dim=1)
        nei_reward_pred_seq = torch.stack(nei_reward_pred_seq, dim=1)
        return nei_move_pred_seq, nei_reward_pred_seq

    def move_single_step(self, fea_query, fea_nei):
        nei_size = fea_nei.shape[-2]
        fea_query = fea_query.unsqueeze(1).repeat(1, nei_size, 1)
        fea_nei = torch.cat((fea_nei, fea_query), dim=-1)
        fea_nei = self.unit_fea_mlp_2(fea_nei)
        fea_nei = F.relu(fea_nei)
        fea_nei = F.dropout(fea_nei, self.p_dropout)
        nei_pred_cls = self.unit_fea_mlp_cls(fea_nei).squeeze(dim=-1)  # dense to 1 then squeeze
        nei_pred_reg = self.unit_fea_mlp_reg(fea_nei).squeeze(dim=-1)  # dense to 1 then squeeze
        return nei_pred_cls, nei_pred_reg

    def update_query(self, query, pos, neg):
        query = self.query_upd_mlp(torch.cat((query, pos, neg), dim=-1))
        query = F.relu(query)
        query = F.dropout(query, self.p_dropout)
        return query

    def aggregate(self, _a, _s, _fea_emb):
        """
        Aggregation function in batch
        :param _a: Adjacency matrix [B, N, N] or [B, A, N, N] where A is dim of neighbors
        :param _s: State vector [B, N, ] or [B, A, N]
        :param _fea_emb: Feature embedding [B, N, d] or [B, A, N, d]
        :return: Aggregated feature of _s given _a
        """
        batch_size, nei_size = _a.shape[0], _a.shape[1]
        device = _a.device
        _fea = torch.einsum('band,ban->bad', _fea_emb, _s) * self.alpha
        cover = torch.ones(batch_size, nei_size, 1, device=device)  # initialize TODO check
        for i in range(self.step_num):
            neis_leq_i_step = torch.einsum('baio,bai->bao', _a, _s)
            cover = ((_s > 0) | (neis_leq_i_step > 0)).to(torch.float)
            new_explore = cover - _s
            _s = cover
            num_new_nodes = new_explore.sum().item()
            if num_new_nodes:
                weight_sum = torch.einsum('band,ban->bad', _fea_emb, new_explore) * (self.alpha ** (i + 2))
                _fea += weight_sum
        _fea /= cover.sum(dim=-1, keepdim=True)
        return _fea

    def bert_encode(self, text_ids):
        bert_output, _ = self.bert_encoder(text_ids)
        return bert_output

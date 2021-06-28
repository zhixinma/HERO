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
        self.unit_fea_mlp_3 = nn.Linear(d_fea, 1)
        self.p_dropout = 0.9

        self.encode = "bert"
        self.freeze_bert = True
        if self.encode == "bert":
            self.bert_encoder = BertModel.from_pretrained("bert-base-uncased")
            self.text_encode_func = self.bert_encode
            for p in self.bert_encoder.parameters():
                p.requires_grad = not self.freeze_bert

    def forward(self, query_tok_ids, _a_nei, _vec_nei, _fea_emb, _nei_mask):
        """
        :param query_tok_ids:
        :param _a_nei: adjacent matrix of neighbors including
        the current state. [B, NEI, N, N]
        :param _vec_nei: [B, NEI]
        :param _fea_emb: [B, N, D]
        :param _nei_mask:
        :return:
        """
        query_fea = self.bert_encode(query_tok_ids)
        move_pred = self.move(query_fea, _a_nei, _vec_nei, _fea_emb, _nei_mask)
        return move_pred

    def move(self, query_fea, _a_nei, _nei, _fea_emb, _nei_mask):
        nei_size = _a_nei.shape[1]
        nei_fea = self.aggregate(_a_nei, _nei, _fea_emb)
        nei_fea = torch.where(_nei_mask > 0, nei_fea, _nei_mask)  # remove nan

        nei_fea = F.relu(nei_fea)
        nei_fea = self.unit_fea_mlp_1(nei_fea)
        nei_fea = F.relu(nei_fea)
        nei_fea = F.dropout(nei_fea, self.p_dropout)

        query_fea = query_fea.mean(dim=-2)
        query_fea = query_fea.unsqueeze(1).repeat(1, nei_size, 1)
        nei_fea = torch.cat((nei_fea, query_fea), dim=-1)
        nei_fea = self.unit_fea_mlp_2(nei_fea)
        nei_fea = F.relu(nei_fea)
        nei_fea = F.dropout(nei_fea, self.p_dropout)

        nei_fea = self.unit_fea_mlp_3(nei_fea)
        return nei_fea

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

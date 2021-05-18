import torch
from torch import nn


class IVCML(nn.Module):
    def __init__(self, d_fea):
        super().__init__()
        self.d_fea = d_fea
        self.mlp_fea = nn.Linear(d_fea, d_fea)

    def forward(self):
        return 0

    def aggregate(self, _a, _s, _fea_emb, alpha, step_num):
        """
        Aggregation function in batch
        :param _a: Adjacency matrix [B, N, N]
        :param _s: State vector [B, N, ]
        :param _fea_emb: Feature embedding [N, d]
        :param alpha: decay rate
        :param step_num: number of steps to aggregate
        :return: Aggregated feature of _s given _a
        """
        batch_size = _a.shape[0]
        _fea = torch.einsum('nd,bn->bd', _fea_emb, _s) * alpha
        cover = torch.ones(batch_size, 1)  # initialize
        for i in range(step_num):
            neis_leq_i_step = torch.einsum('bio,bi->bo', _a, _s)
            cover = ((_s > 0) | (neis_leq_i_step > 0)).to(torch.float)
            new_explore = cover - _s
            _s = cover
            num_new_nodes = new_explore.sum().item()
            if num_new_nodes:
                weight_sum = torch.einsum('nd,bn->bd', _fea_emb, new_explore) * (alpha ** (i + 2))
                _fea += weight_sum
        _fea /= cover.sum(dim=-1, keepdim=True)
        return _fea

    def move(self, emb_nodes, fea):
        self.mlp_fea(fea) * emb_nodes

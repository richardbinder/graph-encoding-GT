import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder)
from torch import nn


@register_node_encoder('LPCAEnc')
class LPCAEncoder(torch.nn.Module):
    def __init__(self, emb_dim, expand_x=False):
        super().__init__()
        
        enc_cfg = cfg.posenc_LPCAEnc
        self.enc_dim = enc_cfg.dim_pe
        self.emb_dim = emb_dim
        self.pass_as_var = enc_cfg.pass_as_var if hasattr(enc_cfg, "pass_as_var") else False
        self.dim_in = enc_cfg.dim_in if hasattr(enc_cfg, "dim_in") else 0

        self.expand_x = expand_x and self.emb_dim - self.enc_dim > 0
        if self.expand_x:
            self.linear_x = nn.Linear(self.dim_in, self.emb_dim - self.enc_dim)

        print("expand_x", self.expand_x)

        
    def forward(self, batch):
        lpca_enc = getattr(batch, 'lpca_enc')

        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x

        if self.enc_dim > 0:
            batch.x = torch.cat((h, lpca_enc), 1)
            assert batch.x.shape[1] == self.emb_dim

        if self.pass_as_var:
            # calculate the adjacency matrix scores
            lpca_adj = []
            k = lpca_enc.shape[1] // 2
            for i in range(batch.batch.max().item() + 1):
                node_mask = batch.batch == i
                enc = lpca_enc[node_mask] 

                L, R = enc[:, :k], enc[:, k:]
                adj_i = L @ R.T

                assert adj_i.shape[0] == batch.x[node_mask].shape[0]

                lpca_adj.append(adj_i)
            setattr(batch, 'lpca_adj', torch.block_diag(*lpca_adj))
        return batch

import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder)
from torch import nn
from torch_geometric.nn.models import MLP


@register_node_encoder('LPCAEnc')
class LPCAEncoder(torch.nn.Module):
    def __init__(self, emb_dim, expand_x=False):
        super().__init__()
        
        enc_cfg = cfg.posenc_LPCAEnc
        self.enc_dim = enc_cfg.dim_pe
        self.emb_dim = emb_dim
        self.pass_as_var = enc_cfg.pass_as_var if hasattr(enc_cfg, "pass_as_var") else False
        self.dim_in = enc_cfg.dim_in if hasattr(enc_cfg, "dim_in") else 0

        pecfg = cfg.posenc_LPCAEnc
        dim_pe = pecfg.dim_pe  # Size of Laplace PE embedding
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        # model_type = pecfg.model  # Encoder NN model type for PEs
        # n_heads = pecfg.n_heads  # Num. attention heads in Trf PE encoder

        self.pe_encoder = MLP(in_channels=dim_pe, hidden_channels=dim_pe*2, out_channels=dim_pe, num_layers=n_layers,
                           dropout=cfg.posenc_LPCAEnc.dropout, norm=cfg.posenc_LPCAEnc.norm)

        # encoder_layer = nn.TransformerEncoderLayer(d_model=dim_pe*4,
        #                                            nhead=n_heads,
        #                                            batch_first=True)
        # self.pe_transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        #
        # self.pe_encoder2 = MLP(in_channels=dim_pe*4, hidden_channels=dim_pe*8, out_channels=dim_pe, num_layers=n_layers,
        #                    dropout=cfg.posenc_LPCAEnc.dropout, norm=cfg.posenc_LPCAEnc.norm)

        self.expand_x = expand_x and self.emb_dim - self.enc_dim > 0
        if self.expand_x:
            self.linear_x = nn.Linear(self.dim_in, self.emb_dim - self.enc_dim)

        print("expand_x", self.expand_x)

        
    def forward(self, batch):
        lpca_enc = getattr(batch, 'lpca_enc')

        pos_enc = lpca_enc

        pos_enc = self.pe_encoder(pos_enc)
        # pos_enc = self.pe_transformer(pos_enc)
        # pos_enc = self.pe_encoder2(pos_enc)

        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x

        if self.enc_dim > 0:
            batch.x = torch.cat((h, pos_enc), 1)
            assert batch.x.shape[1] == self.emb_dim

        if self.pass_as_var:
            # calculate the adjacency matrix scores
            lpca_adj = []
            k = pos_enc.shape[1] // 2
            for i in range(batch.batch.max().item() + 1):
                node_mask = batch.batch == i
                enc = pos_enc[node_mask]

                L, R = enc[:, :k], enc[:, k:]
                adj_i = L @ R.T

                assert adj_i.shape[0] == batch.x[node_mask].shape[0]

                lpca_adj.append(adj_i)
            setattr(batch, 'lpca_adj', torch.block_diag(*lpca_adj))
        return batch

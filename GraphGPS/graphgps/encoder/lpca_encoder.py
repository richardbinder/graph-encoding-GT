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
        model_type = pecfg.model  # Encoder NN model type for PEs
        if model_type not in ['Transformer', 'DeepSet']:
            raise ValueError(f"Unexpected PE model {model_type}")
        self.model_type = model_type
        n_layers = pecfg.layers  # Num. layers in PE encoder model
        n_heads = pecfg.n_heads  # Num. attention heads in Trf PE encoder

        # activation = nn.ReLU  # register.act_dict[cfg.gnn.act]
        # if model_type == 'Transformer':
        #     # Transformer model for LapPE
        #     encoder_layer = nn.TransformerEncoderLayer(d_model=dim_pe,
        #                                                nhead=n_heads,
        #                                                batch_first=True)
        #     self.pe_encoder = nn.TransformerEncoder(encoder_layer,
        #                                             num_layers=n_layers)
        # else:
        #     # DeepSet model for LapPE
        #     layers = []
        #     if n_layers == 1:
        #         layers.append(activation())
        #     else:
        #         self.linear_A = nn.Linear(dim_pe, 2 * dim_pe)
        #         layers.append(activation())
        #         for _ in range(n_layers - 2):
        #             layers.append(nn.Linear(2 * dim_pe, 2 * dim_pe))
        #             layers.append(activation())
        #         layers.append(nn.Linear(2 * dim_pe, dim_pe))
        #         layers.append(activation())
        #     self.pe_encoder = nn.Sequential(*layers)

        self.pe_encoder = MLP(in_channels=dim_pe, hidden_channels=dim_pe*4, out_channels=dim_pe, num_layers=n_layers,
                           dropout=cfg.posenc_LPCAEnc.dropout, norm=cfg.posenc_LPCAEnc.norm)

        self.expand_x = expand_x and self.emb_dim - self.enc_dim > 0
        if self.expand_x:
            self.linear_x = nn.Linear(self.dim_in, self.emb_dim - self.enc_dim)

        print("expand_x", self.expand_x)

        
    def forward(self, batch):
        lpca_enc = getattr(batch, 'lpca_enc')

        pos_enc = lpca_enc
        # pos_enc = self.linear_A(pos_enc)  # (Num nodes) x dim_pe

        # PE encoder: a Transformer or DeepSet model
        if self.model_type == 'Transformer':
            # pos_enc = self.pe_encoder(src=pos_enc,
            #                           src_key_padding_mask=empty_mask[:, :, 0])
            pass
        else:
            pos_enc = self.pe_encoder(pos_enc)

        # Sum pooling
        # pos_enc = torch.sum(pos_enc, 1, keepdim=False)  # (Num nodes) x dim_pe

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

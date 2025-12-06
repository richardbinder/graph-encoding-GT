import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.encoder import AtomEncoder
from torch_geometric.graphgym.register import register_node_encoder

from graphgps.encoder.ast_encoder import ASTNodeEncoder
from graphgps.encoder.kernel_pos_encoder import (
    RWSENodeEncoder,
    HKdiagSENodeEncoder,
    ElstaticSENodeEncoder,
)
from graphgps.encoder.laplace_pos_encoder import LapPENodeEncoder
from graphgps.encoder.ppa_encoder import PPANodeEncoder
from graphgps.encoder.signnet_pos_encoder import SignNetNodeEncoder
from graphgps.encoder.voc_superpixels_encoder import VOCNodeEncoder
from graphgps.encoder.type_dict_encoder import TypeDictNodeEncoder
from graphgps.encoder.linear_node_encoder import LinearNodeEncoder
from graphgps.encoder.equivstable_laplace_pos_encoder import EquivStableLapPENodeEncoder
from graphgps.encoder.graphormer_encoder import GraphormerEncoder
from graphgps.encoder.MLP_count_encoder import (
    MLPNodeCountEncoder,
    MLPGraphCountEncoder,
    MLPNodeCountEncoderX2,
    NodeCountSum,
)
from graphgps.encoder.MLP_encoder import MLPNodeEncoder
from graphgps.encoder.lpca_encoder import LPCAEncoder
from graphgps.encoder.dummy_edge_encoder import DummyNodeEncoder


def concat_node_encoders(encoder_classes, pe_enc_names):
    """
    A factory that creates a new Encoder class that concatenates functionality
    of the given list of two or three Encoder classes. First Encoder is expected
    to be a dataset-specific encoder, and the rest PE Encoders.

    Args:
        encoder_classes: List of node encoder classes
        pe_enc_names: List of PE embedding Encoder names, used to query a dict
            with their desired PE embedding dims. That dict can only be created
            during the runtime, once the config is loaded.

    Returns:
        new node encoder class
    """

    class Concat2NodeEncoder(torch.nn.Module):
        """Encoder that concatenates two node encoders."""

        enc1_cls = None
        enc2_cls = None
        enc2_name = None

        def __init__(self, dim_emb):
            super().__init__()

            # Special handling for Equiv_Stable LapPE where node feats and PE are not concatenated.
            if cfg.posenc_EquivStableLapPE.enable:
                self.encoder1 = self.enc1_cls(dim_emb)
                self.encoder2 = self.enc2_cls(dim_emb)
                return

            # --- Common PE config lookup ---------------------------------------------------------
            pe_name = self.enc2_name

            # Try PE config from posenc_*, fall back to ctenc_*
            posenc_cfg = getattr(cfg, f"posenc_{pe_name}", None)
            ctenc_cfg = getattr(cfg, f"ctenc_{pe_name}", None)

            if posenc_cfg is not None:
                enc2_dim_pe = posenc_cfg.dim_pe
            elif ctenc_cfg is not None:
                enc2_dim_pe = ctenc_cfg.dim_ct
            else:
                raise ValueError(
                    f"Neither posenc_{pe_name} nor ctenc_{pe_name} found in cfg."
                )

            # --- Decide if PE is stacked on top of h or concatenated ----------------------------
            # Default behavior: PE is concatenated with node features -> enc1 gets (dim_emb - enc2_dim_pe)
            stack_on_h = False

            # Prefer ctenc_* if it has a stack_on_h attribute, else fall back to posenc_*
            for cfg_obj in (ctenc_cfg, posenc_cfg):
                if cfg_obj is not None and hasattr(cfg_obj, "stack_on_h"):
                    stack_on_h = bool(getattr(cfg_obj, "stack_on_h"))
                    break

            # If PE is stacked on top of h, encoder1 sees full dim_emb,
            # otherwise it only gets dim_emb - enc2_dim_pe.
            enc1_dim = dim_emb if stack_on_h else dim_emb - enc2_dim_pe

            # --- Build encoders ------------------------------------------------------------------
            self.encoder1 = self.enc1_cls(enc1_dim)
            self.encoder2 = self.enc2_cls(dim_emb, expand_x=False)

        def forward(self, batch):
            batch = self.encoder1(batch)
            batch = self.encoder2(batch)
            return batch

    class Concat3NodeEncoder(torch.nn.Module):
        """Encoder that concatenates three node encoders."""

        enc1_cls = None
        enc2_cls = None
        enc2_name = None
        enc3_cls = None
        enc3_name = None

        def __init__(self, dim_emb):
            super().__init__()
            # PE dims can only be gathered once the cfg is loaded.
            enc2_dim_pe = (
                getattr(cfg, f"posenc_{self.enc2_name}").dim_pe
                if hasattr(cfg, f"posenc_{self.enc2_name}")
                else getattr(cfg, f"ctenc_{self.enc2_name}").dim_ct
            )
            enc3_dim_pe = (
                getattr(cfg, f"posenc_{self.enc3_name}").dim_pe
                if hasattr(cfg, f"posenc_{self.enc3_name}")
                else getattr(cfg, f"ctenc_{self.enc3_name}").dim_ct
            )

            # INCOMPLETE: add "stack_on_h" functionality
            # dim1 = dim_emb
            # dim2 = dim_emb
            # dim3 = dim_emb

            if (
                hasattr(cfg, f"ctenc_{self.enc2_name}")
                and hasattr(getattr(cfg, f"ctenc_{self.enc2_name}"), "stack_on_h")
                and getattr(cfg, f"ctenc_{self.enc2_name}").stack_on_h == True
            ) and (
                hasattr(cfg, f"posenc_{self.enc3_name}")
                and hasattr(getattr(cfg, f"posenc_{self.enc3_name}"), "stack_on_h")
                and getattr(cfg, f"posenc_{self.enc3_name}").stack_on_h == True
            ):
                dim1 = dim_emb
                dim2 = dim_emb
                dim3 = dim_emb
            elif (
                hasattr(cfg, f"ctenc_{self.enc3_name}")
                and hasattr(getattr(cfg, f"ctenc_{self.enc3_name}"), "stack_on_h")
                and getattr(cfg, f"ctenc_{self.enc3_name}.stack_on_h") == True
            ) and (
                hasattr(cfg, f"posenc_{self.enc2_name}.stack_on_h")
                and hasattr(getattr(cfg, f"posenc_{self.enc2_name}"), "stack_on_h")
                and getattr(cfg, f"posenc_{self.enc2_name}").stack_on_h == True
            ):
                dim1 = dim_emb
                dim2 = dim_emb
                dim3 = dim_emb
            else:
                dim1 = dim_emb - enc2_dim_pe - enc3_dim_pe
                dim2 = dim_emb - enc3_dim_pe
                dim3 = dim_emb

            self.encoder1 = self.enc1_cls(dim1)
            self.encoder2 = self.enc2_cls(dim2, expand_x=False)
            self.encoder3 = self.enc3_cls(dim3, expand_x=False)
            # self.encoder1 = self.enc1_cls(dim_emb - enc2_dim_pe - enc3_dim_pe)
            # self.encoder2 = self.enc2_cls(dim_emb - enc3_dim_pe, expand_x=False)
            # self.encoder3 = self.enc3_cls(dim_emb, expand_x=False)

        def forward(self, batch):
            batch = self.encoder1(batch)
            batch = self.encoder2(batch)
            batch = self.encoder3(batch)
            return batch

    # Configure the correct concatenation class and return it.
    if len(encoder_classes) == 2:
        Concat2NodeEncoder.enc1_cls = encoder_classes[0]
        Concat2NodeEncoder.enc2_cls = encoder_classes[1]
        Concat2NodeEncoder.enc2_name = pe_enc_names[0]
        return Concat2NodeEncoder
    elif len(encoder_classes) == 3:
        Concat3NodeEncoder.enc1_cls = encoder_classes[0]
        Concat3NodeEncoder.enc2_cls = encoder_classes[1]
        Concat3NodeEncoder.enc3_cls = encoder_classes[2]
        Concat3NodeEncoder.enc2_name = pe_enc_names[0]
        Concat3NodeEncoder.enc3_name = pe_enc_names[1]
        return Concat3NodeEncoder
    else:
        raise ValueError(
            f"Does not support concatenation of "
            f"{len(encoder_classes)} encoder classes."
        )


# EDITED: added a factory which adds the encoders that sum WL_full with the composed-encoder embeddings of node label and pe
def add_WLfembed_to_encoders(encoder_classes, pe_enc_names):

    class WLf_sum_encoder(torch.nn.Module):

        composed_enc = None
        WLtree_enc = None

        def __init__(self, dim_emb):
            super().__init__()
            self.comp_enc = self.composed_enc(dim_emb)
            self.wl_enc = self.WLtree_enc(dim_emb)

        def forward(self, batch):
            batch = self.comp_enc(batch)
            batch = self.wl_enc(batch)
            return batch

    # get the main (composed) encoder module
    if (
        len(encoder_classes) == 1
    ):  # composed_encoder should just be the dataset specific (initial node label embedder) encoder
        WLf_sum_encoder.composed_enc = encoder_classes[0]
    else:  # composed encoder should be a concatenation of dataset specific encoders along with some positional encoders
        WLf_sum_encoder.composed_enc = concat_node_encoders(
            encoder_classes, pe_enc_names
        )
    WLf_sum_encoder.WLtree_enc = NodeCountSum

    return WLf_sum_encoder


def expand_x_encoder(encoder_class):


    class Expanded_x(torch.nn.Module):


        def __init__(self, dim_emb):
            super().__init__()
            self.encoder_expanded_x = encoder_class(dim_emb, expand_x=True)
    

        def forward(self, batch):
            return self.encoder_expanded_x(batch)

    return Expanded_x


# Dataset-specific node encoders.
ds_encs = {
    "Atom": AtomEncoder,
    "ASTNode": ASTNodeEncoder,
    "PPANode": PPANodeEncoder,
    "TypeDictNode": TypeDictNodeEncoder,
    "VOCNode": VOCNodeEncoder,
    "LinearNode": LinearNodeEncoder,
    "MLPNodeEnc": MLPNodeEncoder,
}

# Positional Encoding node encoders.
pe_encs = {
    "LapPE": LapPENodeEncoder,
    "RWSE": RWSENodeEncoder,
    "HKdiagSE": HKdiagSENodeEncoder,
    "ElstaticSE": ElstaticSENodeEncoder,
    "SignNet": SignNetNodeEncoder,
    "EquivStableLapPE": EquivStableLapPENodeEncoder,
    "GraphormerBias": GraphormerEncoder,
    "LPCAEnc": LPCAEncoder,
}

# Count Encoding node encoders.
ct_encs = {
    "NodeCountEnc": MLPNodeCountEncoder,
    "GraphCountEnc": MLPGraphCountEncoder,
    "NodeCountEncX2": MLPNodeCountEncoderX2,
    "DummyNode": DummyNodeEncoder,
}

# Concat dataset-specific and PE encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for pe_enc_name, pe_enc_cls in pe_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{pe_enc_name}",
            concat_node_encoders([ds_enc_cls, pe_enc_cls], [pe_enc_name]),
        )

# Combine both LapPE and RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+LapPE+RWSE",
        concat_node_encoders(
            [ds_enc_cls, LapPENodeEncoder, RWSENodeEncoder], ["LapPE", "RWSE"]
        ),
    )

# Combine both SignNet and RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+SignNet+RWSE",
        concat_node_encoders(
            [ds_enc_cls, SignNetNodeEncoder, RWSENodeEncoder], ["SignNet", "RWSE"]
        ),
    )

# Combine GraphormerBias with LapPE or RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+GraphormerBias+LapPE",
        concat_node_encoders(
            [ds_enc_cls, GraphormerEncoder, LapPENodeEncoder],
            ["GraphormerBias", "LapPE"],
        ),
    )
    register_node_encoder(
        f"{ds_enc_name}+GraphormerBias+RWSE",
        concat_node_encoders(
            [ds_enc_cls, GraphormerEncoder, RWSENodeEncoder], ["GraphormerBias", "RWSE"]
        ),
    )

# Concat dataset-specific and count encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for ct_enc_name, ct_enc_cls in ct_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{ct_enc_name}",
            concat_node_encoders([ds_enc_cls, ct_enc_cls], [ct_enc_name]),
        )

# Combine counts with RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for ct_enc_name, ct_enc_cls in ct_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{ct_enc_name}+RWSE",
            concat_node_encoders(
                [ds_enc_cls, ct_enc_cls, RWSENodeEncoder], [ct_enc_name, "RWSE"]
            ),
        )

# Positional encoder + LPCA
register_node_encoder(
    f"RWSE+LPCAEnc",
    concat_node_encoders([RWSENodeEncoder, LPCAEncoder], ["RWSE", "LPCAEnc"]),
)

# Exapnd_x LPCA encoder
register_node_encoder(
    f"LPCAEnc-e",
    expand_x_encoder(LPCAEncoder)
)

# WLtree sum encoders:

# Sum WL with dataset-specific encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+NodeCountSum",
        add_WLfembed_to_encoders([ds_enc_cls], [pe_enc_name]),
    )

# Sum WL with dataset-specific and PE encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for pe_enc_name, pe_enc_cls in pe_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{pe_enc_name}+NodeCountSum",
            add_WLfembed_to_encoders([ds_enc_cls, pe_enc_cls], [pe_enc_name]),
        )

# Sum WL with (ds encoder and) both LapPE and RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    register_node_encoder(
        f"{ds_enc_name}+LapPE+RWSE+NodeCountSum",
        add_WLfembed_to_encoders(
            [ds_enc_cls, LapPENodeEncoder, RWSENodeEncoder], ["LapPE", "RWSE"]
        ),
    )

# Sum WL with dataset-specific and count encoders.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for ct_enc_name, ct_enc_cls in ct_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{ct_enc_name}+NodeCountSum",
            add_WLfembed_to_encoders([ds_enc_cls, ct_enc_cls], [ct_enc_name]),
        )

# Sum WL with (ds and) counts with RWSE positional encodings.
for ds_enc_name, ds_enc_cls in ds_encs.items():
    for ct_enc_name, ct_enc_cls in ct_encs.items():
        register_node_encoder(
            f"{ds_enc_name}+{ct_enc_name}+RWSE+NodeCountSum",
            add_WLfembed_to_encoders(
                [ds_enc_cls, ct_enc_cls, RWSENodeEncoder], [ct_enc_name, "RWSE"]
            ),
        )

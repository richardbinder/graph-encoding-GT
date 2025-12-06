#!/bin/bash

config=$1
enc=$2
seed=$3
enc_dim=$4

export ZINC_LPCA_DATA_DIR="encodings/$enc"


cfg_file=$config
if [[ ! -f "$cfg_file" ]]; then
    echo "ERROR: Config does not exist: $cfg_file"
    exit
fi

enc_file="encodings/$enc/lpca.npz"
if [[ ! -f "$enc_file" ]]; then
    echo "ERROR: Encoding does not exist: $enc_file"
    exit
fi

dataset_dim=""

out_dir="."

if [[ $# -eq 3 ]]; then
    dataset_dim="posenc_LPCAEnc.dim_pe $enc_dim"
fi

vocab_dim=48
embed_dim=$((vocab_dim+enc_dim))

conda run -n HomEnv --live-stream python main.py --cfg $cfg_file --repeat 1 seed $seed out_dir $out_dir name_tag none.enc.og $dataset_dim gt.dim_hidden $embed_dim gnn.dim_inner $embed_dim wandb.use True

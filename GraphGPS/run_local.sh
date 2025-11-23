#!/bin/bash

# $1 is the name of the config file
# $2 is the seed
# $3 is the dimension of the encoding (which is 2*k)

conda activate HomEnv

export ZINC_LPCA_DATA_DIR="encodings/k$3"

cfg_file="configs/ZINC/With_Edge_Features/GTe/+$1.yaml"
if [[ ! -f "$cfg_file" ]]; then
    echo "WARNING: Config does not exist: $cfg_file"
    exit
fi

dataset_dim=""

seed=$2
out_dir="."

if [[ $# -eq 3 ]]; then
    dataset_dim="ctenc_LPCAEnc.dim_ct $3"
fi

python main.py --cfg $cfg_file --repeat 1 seed $seed out_dir $out_dir name_tag none.enc.og $dataset_dim wandb.use False

#!/bin/bash

python="~/anaconda/bin/python"

MODE=$1  # "baseline": No transformation
         # "basic": Basic model
         # "copy": Copy model
         # "rule": Rule-based model
GPU=0  # CUDA device
REV_ENC_DIM=192  # Hidden dim size for neural models
REV_LEN=9999999  # Do not modify
BEAM_SIZE=4  # Beam size for neural models
COPY_MECH="attn_nm"  # Do not modify
COPY_WEIGHT="auto"  # Do not modify
LEARN_RATE=0.001  # Learning rate
N_EPOCHS=30  # Num of epochs

if [ $MODE = "baseline" ]; then
    cmd="$python task_quest_trans.py -rev $MODE"
elif [ $MODE = "rule" ]; then
    cmd="$python task_quest_trans.py -rev $MODE"
else
    cmd="CUDA_VISIBLE_DEVICES=$GPU $python task_quest_trans.py -rev $MODE -rev_enc_dim $REV_ENC_DIM -copy_weight $COPY_WEIGHT -copy_mech $COPY_MECH -rev_len $REV_LEN -beam_size $BEAM_SIZE -learn_rate $LEARN_RATE -n_epochs $N_EPOCHS"
fi

echo "$cmd"
eval $cmd

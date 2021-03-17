#!/bin/bash

WORK_PATH="<INSERT_PATH>"

data_setting='60_sub'

train_split="${WORK_PATH}/NTU/NTU60/xsub/NTU_train_seqs.npy"
train_split_labels="${WORK_PATH}/NTU/NTU60/xsub/NTU_train_labels.npy"
val_split="${WORK_PATH}/NTU/NTU60/xsub/NTU_val_seqs.npy"
val_split_labels="${WORK_PATH}/NTU/NTU60/xsub/NTU_val_labels.npy"

gpus=1
progress_bar_refresh_rate=0
ckpt_period=3
num_workers=9

s_weight_bu=(1 0 0 0 1 1 1 0 0 0 1 1 1 0 1)
s_weight_td=(0 1 0 0 1 0 0 1 1 0 1 0 1 1 1)
s_weight_bu_sim=(0 0 1 0 0 1 0 1 0 1 1 1 0 1 1)
s_weight_td_sim=(0 0 0 1 0 0 1 0 1 1 0 1 1 1 1)

weight_cfg=0 

max_epochs=80
num_layers=1
learning_rate=0.0009
hidden_dim=1024
batch_size=32


seed=0

weight_bu=${s_weight_bu[$weight_cfg]}
weight_td=${s_weight_td[$weight_cfg]}
weight_bu_sim=${s_weight_bu_sim[$weight_cfg]}
weight_td_sim=${s_weight_td_sim[$weight_cfg]}

cfg_name="NTU_BUTD_CON"
NOW=$(date +"%Y-%m-%d_%H-%M-%S")
config="configs/NTU_BUTD_CON.yaml"
exp_name="${NOW}_${cfg_name}_LR${learning_rate}_BS${batch_size}_hidden_dim${hidden_dim}_exp_${weight_bu}${weight_td}${weight_bu_sim}${weight_td_sim}"

exp_dir="${WORK_PATH}/prj_ssl_exps/$exp_name"

path_db="${WORK_PATH}/prj_ssl_dbs_iccv"

python pretrain.py  \
                --config $config \
                --seed $seed \
                --exp_name $exp_name \
                --exp_dir $exp_dir \
                --path_db $path_db \
                --gpus $gpus \
                --max_epochs $max_epochs \
                --num_workers $num_workers \
                --progress_bar_refresh_rate $progress_bar_refresh_rate \
                --ckpt_period $ckpt_period \
                --weight_bu $weight_bu \
                --weight_td $weight_td \
                --weight_bu_sim $weight_bu_sim \
                --weight_td_sim $weight_td_sim \
                --train_split $train_split \
                --train_split_labels $train_split_labels \
                --val_split $val_split \
                --val_split_labels $val_split_labels \
                --num_layers $num_layers \
                --learning_rate $learning_rate \
                --hidden_dim $hidden_dim \
                --batch_size $batch_size \

#!/bin/bash

progress_bar_refresh_rate=10

s_weight_bu=(0 0 1)
s_weight_td=(1 1 1)
s_weight_bu_sim=(0 1 0)
s_weight_td_sim=(0 0 0)
weight_cfg=2

max_epochs=150
num_layers=1

WORK_PATH='<INSERT_PATH>'

gpus=1

seed=$i
weight_bu=${s_weight_bu[$weight_cfg]}
weight_td=${s_weight_td[$weight_cfg]}
weight_bu_sim=${s_weight_bu_sim[$weight_cfg]}
weight_td_sim=${s_weight_td_sim[$weight_cfg]}

cfg_name="UCLA_BUTD_CON"
NOW=$(date +"%Y-%m-%d_%H-%M-%S")
config="configs/${cfg_name}.yaml"
exp_name="${NOW}_${cfg_name}_weightcfg${weight_cfg}_seed${seed}"
exp_dir="${WORK_PATH}/prj_ssl_ucla_exps/$exp_name"

path_db="${WORK_PATH}/prj_ssl_ucla_dbs"

python pretrain.py  \
                --config $config \
                --exp_name $exp_name \
                --exp_dir $exp_dir \
                --gpus $gpus \
                --seed $seed \
                --progress_bar_refresh_rate $progress_bar_refresh_rate \
                --weight_bu $weight_bu \
                --weight_td $weight_td \
                --weight_bu_sim $weight_bu_sim \
                --weight_td_sim $weight_td_sim \
                --max_epochs $max_epochs \
                --num_layers $num_layers \
               > logs/${exp_name}.log 2>&1 &

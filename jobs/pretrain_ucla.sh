

device=1
gpus=1
seed=1

batch_size=64
max_epochs=300
learning_rate=0.001

progress_bar_refresh_rate=0

ucla_configs=(\
    "UCLA_AE"\
    "UCLA_VAE"\
    "UCLA_BU_MOCO"\
    "UCLA_BUTD_MOCO"\
    "UCLA_TD_CON"\
)

WORK_PATH='..'

i=0

cfg_name=${ucla_configs[$i]}

NOW=$(date +"%Y-%m-%d_%H-%M-%S")
config="configs/${cfg_name}.yaml"
exp_name="${NOW}_${cfg_name}_seed${seed}"
exp_dir="${WORK_PATH}/ssl_action_exps_3/$exp_name"

path_db="${WORK_PATH}/ssl_action_dbs"

CUDA_VISIBLE_DEVICES=$device \
nohup \
python pretrain.py  \
                --config $config \
                --exp_name $exp_name \
                --exp_dir $exp_dir \
                --gpus $gpus \
                --seed $seed \
                --progress_bar_refresh_rate $progress_bar_refresh_rate \
                --batch_size $batch_size \
                --max_epochs $max_epochs \
                --epochs $max_epochs \
                --learning_rate $learning_rate \
               > logs/${exp_name}.log 2>&1 &



device=1
gpus=1
batch_size=64
max_epochs=300
learning_rate=0.001

seed=1

progress_bar_refresh_rate=0


NOW=$(date +"%Y-%m-%d_%H-%M-%S")

setting="60_sub"
train_split="../NTU/NTU60/NTU60/xsub/NTU_train_seqs.npy"
train_split_labels="../NTU/NTU60/NTU60/xsub/NTU_train_labels.npy"
val_split="../NTU/NTU60/NTU60/xsub/NTU_val_seqs.npy"
val_split_labels="../NTU/NTU60/NTU60/xsub/NTU_val_labels.npy"

# setting="60_view"
# train_split="../NTU/NTU60/NTU60/xview/NTU_train_seqs.npy"
# train_split_labels="../NTU/NTU60/NTU60/xview/NTU_train_labels.npy"
# val_split="../NTU/NTU60/NTU60/xview/NTU_val_seqs.npy"
# val_split_labels="../NTU/NTU60/NTU60/xview/NTU_val_labels.npy"

# setting="120_sub"
# train_split="../NTU/NTU120/xsub/NTU_train_seqs.npy"
# train_split_labels="../NTU/NTU120/xsub/NTU_train_labels.npy"
# val_split="../NTU/NTU120/xsub/NTU_val_seqs.npy"
# val_split_labels="../NTU/NTU120/xsub/NTU_val_labels.npy"

# setting="120_setup"
# train_split="../NTU/NTU120/xsetup/NTU_train_seqs.npy"
# train_split_labels="../NTU/NTU120/xsetup/NTU_train_labels.npy"
# val_split="../NTU/NTU120/xsetup/NTU_val_seqs.npy"
# val_split_labels="../NTU/NTU120/xsetup/NTU_val_labels.npy"


ntu_configs=(\
    "NTU_AE"\
    "NTU_VAE"\
    "NTU_BU_MOCO"\
    "NTU_BUTD_MOCO"\
    "NTU_TD_CON"\
)

WORK_PATH='..'

seed=1

cfg_name="UCLA_BUTD_MOCO"

NOW=$(date +"%Y-%m-%d_%H-%M-%S")
config="configs/${cfg_name}.yaml"
exp_name="${NOW}_${cfg_name}_${setting}_seed${seed}"
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
                --train_split $train_split \
                --train_split_labels $train_split_labels \
                --val_split $val_split \
                --val_split_labels $val_split_labels \

               > logs/${exp_name}.log 2>&1 &


# Self Supervised Learning of Skeleton Sequences for Action Recognition

## Requirements

1. install pytorch using the command from the official website
2. `pip install -r requirements`
 

## Pretraining

For pretraining and linear classification, choose a configuration from `configs/` and run `pretrain.py`. Overwrite parameters by adding them as arguments. The new configuration is saved as a yaml file in the experiment directory. Experiments are logged using tensorboard by default. If a user `neptune_key` is specified, the experiment is logged in neptune. A summary of the results is saved in a database as a csv file. Use `python join_db.py` to join all csv files into one containing all experiments.

The following scirpt is an example:

```
device=3
gpus=1
batch_size=256
max_epochs=5
learning_rate=0.001

# set to 0 to disable logging
progress_bar_refresh_rate=1

# checkpoint frequency
ckpt_period=3

NOW=$(date +"%Y-%m-%d_%H-%M-%S")
base_cfg="configs/NTU_AE.yaml"
exp_name="${NOW}_NTU_AE"
exp_dir="../prj_ssl_ntu_exps/$exp_name"
path_db="../ssl_db"

mkdir logs

CUDA_VISIBLE_DEVICES=$device \
nohup \
python pretrain.py  \
                --config $base_cfg \
                --exp_name $exp_name \
                --exp_dir $exp_dir \
                --path_db $path_db \
                --gpus $gpus \
                --batch_size $batch_size \
                --max_epochs $max_epochs \
                --epochs $max_epochs \
                --learning_rate $learning_rate \
                --progress_bar_refresh_rate $progress_bar_refresh_rate \
                --ckpt_period $ckpt_period \
                > logs/${exp_name}.log 2>&1 &

```

## Linear Classification

For linear classification of an already trained model, specify the configuration file of the experiment and run the `lincls.py`.

```
device=0

exp_name='2020-11-02_15-40-52_NTU_VAE'
exp_path='<path to experiment>/${exp_name}/config.yaml'

CUDA_VISIBLE_DEVICES=$device \
python lincls.py --config $exp_cfg

```

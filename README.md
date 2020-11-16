# Self Supervised Learning of Skeleton Sequences for Action Recognition


## Results

Current linear classification results:

|                               | NTU sub | UCLA  | NTU xview |
|-------------------------------|---------|-------|-----------|
| cvpr 20 baseline              | 50.7    | 84.9  |  76.3     |
| AutoEncoder (Predict Cluster) | 59.19   | 83.45 |  66.67    |
| VAE                           | 66.93   | 85.0  |  73.35    |
| Moco                          | 66.99   | 83.73 |  71.86    |
| Moco-AE                       | 68.19   | 85.0  |           |
| Moco-AE + latent similarity   | 70.53   | 86.08 |  77.01    |
| Moco-AE + top-down similarity | 67.33   | 84.34 |           |

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


## Hyperparameter Tuning

For tuning hyper parameters, specify a configuration from `configs/` then manually choose the hyperparameters and search space in `tune_ray.py` function `tune_asha`. The script will take all ressources available in the node so make sure spcify the gpus before running it.

```
device="2,3,4,5,6,7" # gpus used by the script
gpus=1 # number of gpus used by a single job 
max_epochs=250
n_tune_samples=200 # number of jobs

NOW=$(date +"%Y-%m-%d_%H-%M-%S")

base_cfg="configs/UCLA_BU_MOCO.yaml"
exp_name="${NOW}_UCLA_BU_MOCO"


exp_dir="/home/azerroug/prj_ssl_ucla_bu_tune/"
path_db="/home/azerroug/prj_ssl_ucla_bu_tune/dbs"


CUDA_VISIBLE_DEVICES=$device \
nohup \
python tune_ray.py  \
                --config $base_cfg \
                --exp_dir $exp_dir \
                --path_db $path_db \
                --gpus $gpus \
                --max_epochs $max_epochs \
                --epochs $max_epochs \
                --n_tune_samples $n_tune_samples \
                > logs/${exp_name}.log 2>&1 &

# use --smoke-test for debugging
```





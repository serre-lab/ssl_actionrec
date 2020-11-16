#!/bin/bash
#SBATCH --mem=30G
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -J NTU
#SBATCH -p gpu 
#SBATCH -o logs/%x_%A_%a_%J.out
#SBATCH -e logs/%x_%A_%a_%J.err
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=carney-tserre-condo

# module load anaconda/3-5.2.0
# source activate t16

device=7

# exp_cfg='../prj_ssl_ntu_exps/2020-10-23_08-42-08_ucla_test/config.yaml'
# exp_cfg='../prj_ssl_ntu_exps/2020-10-29_12-39-35_NTU_BU_MOCO/config.yaml'
# exp_cfg='../prj_ssl_ntu_exps/2020-10-28_08-24-54_NTU_BU_MOCO/config.yaml'
# exp_cfg='../prj_ssl_ntu_exps/2020-10-27_11-56-20_NTU_AE/config.yaml'
# exp_cfg='../prj_ssl_ntu_exps/2020-10-27_10-06-58_NTU_BUTD_MOCO/config.yaml'

# exp_cfg='../prj_ssl_ntu_exps/2020-10-29_08-42-42_UCLA_AE/config.yaml'

# exp_cfg='../prj_ssl_ntu_exps/2020-10-31_09-03-59_NTU_BUTD_MOCO/config.yaml'
# exp_cfg='../prj_ssl_ntu_exps/2020-10-31_12-22-11_NTU_BU_MOCO/config.yaml'

# exp_cfg='../prj_ssl_ntu_exps/2020-11-02_03-36-07_NTU_BUTD_MOCO/config.yaml'

# exp_cfg='../prj_ssl_ntu_exps/2020-11-02_13-22-10_NTU_VAE/config.yaml'

# exp_cfg='../prj_ssl_ntu_exps/2020-11-02_15-40-52_NTU_VAE/config.yaml'

# exp_cfg='../prj_ssl_ntu_exps/2020-10-25_19-15-09_UCLA_BU_MOCO/config.yaml'


# CUDA_VISIBLE_DEVICES=$device \
# python lincls.py  --config $exp_cfg

exps=(\
    # "2020-11-11_04-21-40_NTU_BUTD_MOCO_QOnly"\
    # "2020-11-11_04-22-16_NTU_BUTD_MOCO_QOnly"\
    # "2020-11-11_04-22-55_NTU_BUTD_MOCO_QOnly"\
    # "2020-11-11_05-05-50_NTU_BUTD_MOCO_QOnly"\
    # "2020-11-12_09-22-20_an_UCLA_BU_MOCO"\
    # "2020-11-12_09-22-20_an_UCLA_BUTD_MOCO_LS"\
    # "2020-11-12_09-48-04_an_UCLA_VAE"\
    # "prj_ssl_exps_120/2020-11-13_18-16-15_NTU_TD_CON_120_sub_seed1"\
    # "prj_ssl_exps_120/2020-11-13_18-16-32_NTU_TD_CON_120_setup_seed1"\
    # "prj_ssl_exps_120/2020-11-14_07-30-28_NTU_AE_120_sub_seed1"\
    # "prj_ssl_exps_120/2020-11-14_07-30-28_NTU_BU_MOCO_120_sub_seed1"\
    # "prj_ssl_exps_120/2020-11-14_07-37-13_NTU_BUTD_MOCO_120_sub_seed1"\
    # "prj_ssl_exps_120/2020-11-14_07-37-13_NTU_BUTD_MOCO_LS_120_sub_seed1"\
    # "2020-11-12_17-25-04_an_NTU_AE"\
    # "2020-11-12_17-25-04_an_NTU_VAE"\
    "2020-11-12_17-25-04_an_NTU_BU_MOCO"\
    "2020-11-12_17-25-04_an_NTU_BUTD_MOCO"\
    "2020-11-12_17-25-04_an_NTU_BUTD_MOCO_LS"\
    # "2020-11-14_10-53-51_NTU_BUTD_MOCO_LS_120_setup_seed1"\
    # "2020-11-14_10-53-51_NTU_BUTD_MOCO_120_setup_seed1"\
)



 
# sed -i 's/\/tmpdir\/zerroug\/NTU/\/media\/data_cifs_lrs\/projects\/prj_ssl_ntu/g' 


# setting="120_sub"
# train_split="/media/data_cifs_lrs/projects/prj_ssl_ntu/NTU120/xsub/NTU_train_seqs.npy"
# train_split_labels="/media/data_cifs_lrs/projects/prj_ssl_ntu/NTU120/xsub/NTU_train_labels.npy"
# val_split="/media/data_cifs_lrs/projects/prj_ssl_ntu/NTU120/xsub/NTU_val_seqs.npy"
# val_split_labels="/media/data_cifs_lrs/projects/prj_ssl_ntu/NTU120/xsub/NTU_val_labels.npy"

# setting="120_setup"
# train_split="/media/data_cifs_lrs/projects/prj_ssl_ntu/NTU120/xsetup/NTU_train_seqs.npy"
# train_split_labels="/media/data_cifs_lrs/projects/prj_ssl_ntu/NTU120/xsetup/NTU_train_labels.npy"
# val_split="/media/data_cifs_lrs/projects/prj_ssl_ntu/NTU120/xsetup/NTU_val_seqs.npy"
# val_split_labels="/media/data_cifs_lrs/projects/prj_ssl_ntu/NTU120/xsetup/NTU_val_labels.npy"


# "2020-11-14_07-37-13_NTU_BUTD_MOCO_LS_120_sub_seed1"\
    # "2020-11-13_10-03-28_NTU_BUTD_MOCO_120_sub_seed10"\
    
# exps=(\
#     # "2020-11-13_01-23-32_NTU_AE_120_sub_seed1"\
#     # "2020-11-13_03-21-11_NTU_AE_120_sub_seed10"\
#     # "2020-11-14_07-30-28_NTU_AE_120_sub_seed1"\
#     # "2020-11-13_04-27-06_NTU_VAE_120_sub_seed1"\
#     # "2020-11-13_04-27-06_NTU_VAE_120_sub_seed5"\
#     # "2020-11-13_06-04-20_NTU_VAE_120_sub_seed15"\
#     # "2020-11-13_06-28-54_NTU_BU_MOCO_120_sub_seed1"\
#     # "2020-11-13_07-43-32_NTU_BU_MOCO_120_sub_seed5"\
#     # "2020-11-13_07-43-41_NTU_BU_MOCO_120_sub_seed15"\
#     # "2020-11-13_08-47-44_NTU_BU_MOCO_120_sub_seed20"\
#     # "2020-11-14_07-30-28_NTU_BU_MOCO_120_sub_seed1"\
#     # "2020-11-14_07-37-13_NTU_BUTD_MOCO_120_sub_seed1"\
#     # "2020-11-14_11-26-29_NTU_BUTD_MOCO_LS_120_sub_seed1"\
#     # "2020-11-13_18-16-15_NTU_TD_CON_120_sub_seed1"\
#     # #
#     # "2020-11-13_19-57-26_NTU_AE_120_setup_seed1"\
#     # "2020-11-13_20-37-26_NTU_AE_120_setup_seed5"\
#     # "2020-11-13_21-34-39_NTU_AE_120_setup_seed10"\
#     # "2020-11-14_11-25-20_NTU_AE_120_setup_seed1"\
#     # "2020-11-13_23-15-32_NTU_VAE_120_setup_seed1"\
#     # "2020-11-13_23-37-56_NTU_VAE_120_setup_seed10"\
#     # "2020-11-13_23-37-56_NTU_VAE_120_setup_seed5"\
#     # "2020-11-14_00-40-15_NTU_VAE_120_setup_seed15"\
#     # "2020-11-14_00-52-19_NTU_BU_MOCO_120_setup_seed1"\
#     # "2020-11-14_01-12-54_NTU_BU_MOCO_120_setup_seed10"\
#     # "2020-11-14_01-14-26_NTU_BU_MOCO_120_setup_seed15"\
#     # "2020-11-14_02-20-42_NTU_BU_MOCO_120_setup_seed20"\
#     # "2020-11-14_03-11-33_NTU_BUTD_MOCO_120_setup_seed1"\
#     # "2020-11-14_03-28-29_NTU_BUTD_MOCO_120_setup_seed15"\
#     # "2020-11-14_10-53-51_NTU_BUTD_MOCO_120_setup_seed1"\
#     # "2020-11-14_10-53-51_NTU_BUTD_MOCO_LS_120_setup_seed1"\
#     # "2020-11-13_18-16-32_NTU_TD_CON_120_setup_seed1"\
# )



devices=(0 1 2 3 4 5 7)

i=2

# for i in {$idx..$idx}; do
device=${devices[$i]}
# cfg_dir="../prj_ssl_exps_120/${exp}/config.yaml"
exp=${exps[$i]}
# exp=${exps[0]}
# exp_dir="../prj_ssl_exps_120/${exp}"
exp_dir="../prj_ssl_ntu_exps_2/${exp}"
cfg="${exp_dir}/config.yaml"

# path_db="../prj_ssl_dbs_120_2"
path_db="../ssl_db"


CUDA_VISIBLE_DEVICES=$device \
nohup \
python lincls_mlp.py  --config $cfg \
                --path_db $path_db \
    > logs/linc_cls_mlp_${exp}.log 2>&1 &
                # --exp_dir $exp_dir \

# done


# ############# 60_view results
# # lincls
# results={
#     "AE_120_sub":[48.97],
#     "VAE_120_sub":[56.27, 56.29],
#     "BU_MOCO_120_sub":[57.18, 56.94, 56.75, 57.26, 54.83],
#     "BUTD_MOCO_120_sub":[56.98, 56.95],
#     "BUTD_MOCO_LS_120_sub":[59.17],
#     "BUTD_MOCO_Q_120_sub":[],
#     "BUTD_MOCO_RS_120_sub":[],
# }
# results={
#     "AE_120_sub":[0.4496],
#     "VAE_120_sub":[0.47, 0.47],
#     "BU_MOCO_120_sub":[0.45, 0.46, 0.46, 0.46, 0.4501],
#     "BUTD_MOCO_120_sub":[0.4532, 0.4553],
#     "BUTD_MOCO_LS_120_sub":[0.4597],
#     "BUTD_MOCO_Q_120_sub":[],
#     "BUTD_MOCO_RS_120_sub":[],
# }


# # lincls
# results={
#     "AE_120_setup": [49.35],
#     "VAE_120_setup":[57.19, 59.24, 57.85, 52.59],
#     "BU_MOCO_120_setup":	[59.89,60.25,59.53,60.25],
#     "BUTD_MOCO_120_setup":	[59.56],
#     "BUTD_MOCO_LS_120_setup":	[61.55],
#     "BUTD_MOCO_Q_120_setup":	[],
#     "BUTD_MOCO_RS_120_setup":	[],
# }
# results={
#     "AE_120_setup": [0.47982],
#     "VAE_120_setup":[0.49, 0.53, 0.52, 0.45],
#     "BU_MOCO_120_setup":	[0.5, 0.51, 0.5, 0.51],
#     "BUTD_MOCO_120_setup":	[0.5023],
#     "BUTD_MOCO_LS_120_setup":	[0.5035],
#     "BUTD_MOCO_Q_120_setup":	[],
#     "BUTD_MOCO_RS_120_setup":	[],
# }


# {

# "AE_120_sub_seed1": [0.4496, 48.96],
# "AE_120_sub_seed10": [0.4643, 48.63],

# "VAE_120_sub_seed1": [0.4704, 56.17],
# "VAE_120_sub_seed5": [0.4648, 56.28],

# "BU_MOCO_120_sub_seed1": [0.4461, 57.18],
# "BU_MOCO_120_sub_seed15": [0.4491, 56.74],

# "BUTD_MOCO_120_sub_seed1": [0.4543, 54.20],

# "BUTD_MOCO_LS_120_sub_seed1": [0.4551, 59.49],
# "TD_CON_120_sub_seed1": [0.4124, 55.10],


# "AE_120_setup_seed1": [0.5006, 51.92],
# "AE_120_setup_seed5": [0.4874, 50.63],
# "AE_120_setup_seed10": [0.5066, 52.85],

# "VAE_120_setup_seed1": [0.4616, 55.51],
# "VAE_120_setup_seed10": [0.5162, 59.24],
# "VAE_120_setup_seed15": [0.4482, 52.59],

# "BU_MOCO_120_setup_seed1": [0.4904, 45.35],
# "BU_MOCO_120_setup_seed10": [0.4963, 48.14],
# "BU_MOCO_120_setup_seed15": [0.4918, 59.50],
# "BU_MOCO_120_setup_seed20": [0.4924, 59.94],

# "BUTD_MOCO_120_setup_seed1": [0.5023, 57.78],
# "BUTD_MOCO_120_setup_seed15": [0.5006, 57.17],

# "BUTD_MOCO_LS_120_setup_seed1": [0.5035, 61.53],
# "TD_CON_120_setup_seed1": [0.4700, 58.47],

# }


# {
# "AE_120_sub_seed1": 48.96,
# "AE_120_sub_seed10": 48.63,
# "VAE_120_sub_seed1": 56.17,
# "VAE_120_sub_seed5": 56.28,
# "BU_MOCO_120_sub_seed1": 57.18,
# "BU_MOCO_120_sub_seed15": 56.74,
# "BUTD_MOCO_120_sub_seed1": 54.20,
# "BUTD_MOCO_LS_120_sub_seed1": 59.49,
# "TD_CON_120_sub_seed1": 55.10,
# "AE_120_setup_seed1": 51.92,
# "AE_120_setup_seed5": 50.63,
# "AE_120_setup_seed10": 52.85,
# "VAE_120_setup_seed1": 55.51,
# "VAE_120_setup_seed10": 59.24,
# "VAE_120_setup_seed15": 52.59,
# "BU_MOCO_120_setup_seed1": 45.35,
# "BU_MOCO_120_setup_seed10": 48.14,
# "BU_MOCO_120_setup_seed15": 59.50,
# "BU_MOCO_120_setup_seed20": 59.94,
# "BUTD_MOCO_120_setup_seed1": 57.78,
# "BUTD_MOCO_120_setup_seed15": 57.17,
# "BUTD_MOCO_LS_120_setup_seed1": 61.53,
# "TD_CON_120_setup_seed1": 58.47,
# }

# {
# "AE_120_sub_seed1": 0.4496,
# "AE_120_sub_seed10": 0.4643,
# "VAE_120_sub_seed1": 0.4704,
# "VAE_120_sub_seed5": 0.4648,
# "BU_MOCO_120_sub_seed1": 0.4461,
# "BU_MOCO_120_sub_seed15": 0.4491,
# "BUTD_MOCO_120_sub_seed1": 0.4543,
# "BUTD_MOCO_LS_120_sub_seed1": 0.4551,
# "TD_CON_120_sub_seed1": 0.4124,
# "AE_120_setup_seed1": 0.5006,
# "AE_120_setup_seed5": 0.4874,
# "AE_120_setup_seed10": 0.5066,
# "VAE_120_setup_seed1": 0.4616,
# "VAE_120_setup_seed10": 0.5162,
# "VAE_120_setup_seed15": 0.4482,
# "BU_MOCO_120_setup_seed1": 0.4904,
# "BU_MOCO_120_setup_seed10": 0.4963,
# "BU_MOCO_120_setup_seed15": 0.4918,
# "BU_MOCO_120_setup_seed20": 0.4924,
# "BUTD_MOCO_120_setup_seed1": 0.5023,
# "BUTD_MOCO_120_setup_seed15": 0.5006,
# "BUTD_MOCO_LS_120_setup_seed1": 0.5035,
# "TD_CON_120_setup_seed1": 0.4700,
# }

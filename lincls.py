import os
import sys
import yaml
import argparse

from typing import Union
from warnings import warn

import pandas as pd

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from pytorch_lightning.loggers.neptune import NeptuneLogger 
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from pl_bolts.metrics import precision_at_k


import modules
import datasets

def parse_args(parser, argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args, _ = parser.parse_known_args(argv)
    
    config_vars = {}
    if args.config is not None:
        with open(args.config, 'r') as stream:
            config_vars = yaml.load(stream, Loader=yaml.FullLoader)
            
        default_args = argparse.Namespace()
        default_args.__dict__.update(config_vars)
        new_keys = {}
        for k, v in args.__dict__.items():
            if '--'+k in argv or '-'+k in argv or (k not in default_args):
                new_keys[k] = v

        default_args.__dict__.update(new_keys)
        args = default_args
    
    return args



def lincls(args, model):

    # extract dataset features with the model
    dataset_type = vars(datasets)[args.dataset]
    data_args = args.__dict__
    data_args.update({'train_transforms': None, 'val_transforms': None})
    datamodule = dataset_type(**data_args)
    
    train_loader = datamodule.train_dataloader(shuffle=False, drop_last=False)
    val_loader = datamodule.val_dataloader()

    encoder = model.get_encoder().cuda()
    encoder.eval()

    train_features = []
    train_target = []
    with torch.no_grad():
        for batch, target in train_loader:
            input_, seq_len = batch 
            train_features.append(encoder(input_.cuda(), seq_len).cpu())
            train_target.append(torch.Tensor(target))

    train_features = torch.cat(train_features, 0)
    train_target = torch.cat(train_target, 0)

    # train_features = torch.randn([20000, args.hidden_dim])
    # train_target = torch.randint(10, [20000])

    val_features = []
    val_target = []
    with torch.no_grad():
        for batch, target in val_loader:
            input_, seq_len = batch 
            val_features.append(encoder(input_.cuda(), seq_len).cpu())
            val_target.append(torch.Tensor(target))
    
    val_features = torch.cat(val_features, 0)
    val_target = torch.cat(val_target, 0)
    
    # val_features = torch.randn([1000, args.hidden_dim])
    # val_target = torch.randint(10, [1000])
    
    batch_size = 512 if 'NTU' in args.dataset else 128
    # batch_size = 128
    print(datamodule.num_classes)
    datamodule = datasets.FeatureDataModule(train_features, train_target, val_features, val_target, num_workers=4, batch_size=batch_size)
    
    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=os.path.join(args.exp_dir,'lincls'), save_top_k=1, mode='max', monitor='val_acc1_agg', period=1) # , filename='{epoch}-{knn_acc}' 

    # trainer = pl.Trainer(max_epochs = 100, progress_bar_refresh_rate=0, weights_summary=None, gpus=1) #, logger=logger, checkpoint_callback=model_checkpoint
    trainer = pl.Trainer(max_epochs = 70, weights_summary=None, gpus=1, checkpoint_callback=model_checkpoint, progress_bar_refresh_rate=0)
    
    model = modules.LinearClassifierMod(input_dim = args.hidden_dim,
                                        n_label = datamodule.num_classes,
                                        learning_rate = 2,
                                        momentum = 0.9,
                                        weight_decay = 0,
                                        epochs = 70,
                                        lr_decay_rate = 0.01
                                        )

    trainer.fit(model, datamodule)
    
    # best_ckpt = trainer.checkpoint_callback.best_model_path

    ckpts = list(filter(lambda x:'lincls' in x and '.ckpt' in x, os.listdir(args.exp_dir)))
    best_ckpt = ckpts[-1] if len(ckpts) == 1 else ckpts[-2]
    best_ckpt = os.path.join(args.exp_dir, best_ckpt)    
    # classifier = modules.LinearClassifierMod.load_from_checkpoint(checkpoint_path=best_ckpt)
    print(best_ckpt)
    best_model = modules.LinearClassifierMod.load_from_checkpoint(checkpoint_path=best_ckpt)
    
    result_dict = {}

    # lincls_result = trainer.test(model=best_model, verbose=True)[0]
    lincls_result = trainer.test(model=best_model, datamodule=datamodule)[0]


    ##########################################################################################
    # # baseline
    # classifier = best_model.classifier
    # # val_loader = datamodule.val_dataloader()
    # val_logits_trials = []
        
    # val_logits = []
    # val_targets = []
    # with torch.no_grad():
    #     for batch, target in val_loader:
    #         input_, seq_len = batch 
    #         val_logits.append(classifier(encoder(input_.cuda(), seq_len)).cpu())
    #         val_targets.append(torch.Tensor(target))
    #         print(target[-1])
    
    # base_val_logits = torch.cat(val_logits, 0)
    # val_targets = torch.cat(val_targets, 0)
    
    # print('logit', base_val_logits[0])

    # acc1, acc5 = precision_at_k(base_val_logits, val_targets, top_k=(1, 5))
    
    # print('acc', acc1, acc5)
    # # results['baseline'] = np.array([acc1, acc5])[None,:]
    # # base_val_logits = base_val_logits.numpy()
    ##########################################################################################

    # print('best_ckpt', best_ckpt)
    # print('encoder weight')
    # print(encoder.encoder.rnn.bias_hh_l0[:10])
    # print('classifier weight')
    # print(best_model.classifier.weight[:5])

    result_dict.update(lincls_result)
    
    knn_acc = modules.knn(train_features.numpy(), val_features.numpy(), train_target.numpy(), val_target.numpy(), nn=1)
    result_dict['knn_acc'] = knn_acc

    return result_dict

def cli_main():
    
    argv = sys.argv[1:]
    # argv = ['--config',     'configs/base.yaml',
    #         '--exp_name',   'test',
    #         '--exp_dir',    '../prj_ssl_ntu_exps/test']

    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c", "--config", default=None, help="where to load YAML configuration", metavar="FILE")
    parser.add_argument('--exp_dir', type=str, default=None, help='experiment output directory')
    parser.add_argument('--path_db', type=str, default='../dbs', help='neptune project directory')

    args = parser.parse_args(argv)

    new_exp_dir=args.exp_dir
    new_path_db=args.path_db
    
    with open(args.config, 'r') as stream:
        config_vars = yaml.load(stream, Loader=yaml.FullLoader)
        
    args = argparse.Namespace()
    args.__dict__.update(config_vars)
    
    if new_exp_dir is not None:
        args.exp_dir=new_exp_dir

    if new_path_db is not None:
        args.path_db=new_path_db
    
    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    
    # get model and model args
    model_type = vars(modules)[args.model] 
    
    # get dataset and dataset args
    dataset_type = vars(datasets)[args.dataset]

    # save config
    with open(os.path.join(args.exp_dir, 'config.yaml'), 'w') as cfg_file:
        yaml.dump(args.__dict__, cfg_file)

    if args.neptune_key != '':
        logger = NeptuneLogger(
            api_key=args.neptune_key,
            project_name=args.neptune_project,
            close_after_fit=False,
            experiment_name=args.exp_name,  # Optional,
            params=args.__dict__, # Optional,
            tags=["pl"],  # Optional,
            # upload_stderr=False,
            # upload_stdout=False
        )
    else:
        logger = TensorBoardLogger(args.exp_dir) #, name="my_model"
    
    # ckpt = list(filter(lambda x:'.ckpt' in x, os.listdir(args.exp_dir)))[-1]
    # ckpt = os.path.join(args.exp_dir, ckpt)
    
    ckpts = list(filter(lambda x:'epoch=' in x, os.listdir(args.exp_dir)))
    
    best_epoch = max([int( x.replace('epoch=','').replace('.ckpt','')) for x in ckpts])
    best_ckpt = os.path.join(args.exp_dir, 'epoch=' + str(best_epoch) + '.ckpt')
    model = model_type.load_from_checkpoint(best_ckpt)

    lincls_results = lincls(args, model)

    print(best_ckpt)
    
    
    print('test results')
    for k,v in lincls_results.items():
        print(k, v)

    db_path = os.path.join(args.path_db, args.exp_name + '_db.csv')
    
    if os.path.exists(db_path):
        df = pd.read_csv(db_path, index_col=0)
    else:
        df = pd.DataFrame()
    
    output_dict = {
        'exp_name': args.exp_name,
        'exp_dir': args.exp_dir,
        'model': args.model,
        'dataset': args.dataset,
    }

    output_dict.update(lincls_results)
    
    if args.neptune_key != '':
        for k, v in pretrain_result.items():
            logger.experiment.log_metric(k, v)
            
        for k, v in lincls_results.items():
            logger.experiment.log_metric(k, v)

    df = df.append(output_dict, ignore_index=True)
    df.to_csv(db_path)


if __name__ == '__main__':
    cli_main()

    # results = lincls_test()
    # print(results)


# results={
#     "TD_Con_UCLA": 85.50
#     "TD_Con_60_sub": 65.13
#     "TD_Con_60_view": 69.94
#     "TD_Con_120_sub": 55.11
#     "TD_Con_120_setup": 58.45
# }

# results={
#     "TD_Con_UCLA": 0.7956
#     "TD_Con_60_sub": 0.5278
#     "TD_Con_60_view": 0.6089
#     "TD_Con_120_sub": 0.4124
#     "TD_Con_120_setup": 0.4700
# }
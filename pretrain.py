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
from pytorch_lightning.callbacks import Callback

import modules
import datasets

def parse_args(parser, argv=None):

    if argv is None:
        argv = sys.argv[1:]

    args, _ = parser.parse_known_args(argv)
    # args = parser.parse_args(argv)
    
    config_vars = {}
    
    if args.config is not None:
        with open(args.config, 'r') as stream:
            config_vars = yaml.load(stream, Loader=yaml.FullLoader)
            
        default_args = argparse.Namespace()
        default_args.__dict__.update(args.__dict__)
        default_args.__dict__.update(config_vars)

        new_keys = {}
        for k, v in args.__dict__.items():
            if '--'+k in argv or '-'+k in argv or (k not in default_args):
                new_keys[k] = v

        default_args.__dict__.update(new_keys)
        args = default_args
    
    return args


class KNNEval(Callback):
    def __init__(self, period=1):
        super().__init__()
        self.period = period

    def on_validation_epoch_end(self, trainer, pl_module):
        
        if trainer.running_sanity_check:
            return
            
        epoch = trainer.current_epoch
        if epoch % self.period == 0:
                
            # encoder = pl_module.get_encoder()
            encoder = pl_module.get_representations

            train_features = []
            train_target = []
            with torch.no_grad():
                for batch, target in trainer.datamodule.train_dataloader(transform=None): #(batch_size=512):
                    input_, seq_len = batch 
                    train_features.append(encoder(input_.to(pl_module.device), seq_len).cpu())
                    train_target.append(torch.Tensor(target))

            train_features = torch.cat(train_features, 0)
            train_target = torch.cat(train_target, 0)

            val_features = []
            val_target = []
            with torch.no_grad():
                for batch, target in trainer.datamodule.val_dataloader(transform=None): #(batch_size=512):
                    input_, seq_len = batch 
                    val_features.append(encoder(input_.to(pl_module.device), seq_len).cpu())
                    val_target.append(torch.Tensor(target))
            
            val_features = torch.cat(val_features, 0)
            val_target = torch.cat(val_target, 0)
            
            knn_acc = modules.knn(train_features.numpy(), val_features.numpy(), train_target.numpy(), val_target.numpy(), nn=1)
            
            trainer.logger.log_metrics({'knn_acc': knn_acc}, step=trainer.current_epoch) #
            trainer.logger_connector.callback_metrics.update({'knn_acc': knn_acc})

def cli_main():
    
    argv = sys.argv[1:]


    parser = argparse.ArgumentParser()
    
    parser.add_argument("-c", "--config", default=None, help="where to load YAML configuration", metavar="FILE")

    parser.add_argument('--exp_name', type=str, default='test', help='experiment name')
    parser.add_argument('--exp_dir', type=str, default='../experiments/', help='experiment output directory')
    parser.add_argument('--path_db', type=str, default='../dbs', help='neptune project directory')

    parser.add_argument('--model', type=str, default='MocoV2', help='self supervised training method')
    parser.add_argument('--dataset', type=str, default='NTU_SSL', help='dataset to use for training')
    
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    
    parser.add_argument('--resume_training', action='store_true', help='resume training from checkpoint training')
    
    
    args = parse_args(parser, argv)
    
    if args.seed is not None:
        pl.seed_everything(args.seed)
    
    # trainer args
    parser = pl.Trainer.add_argparse_args(parser)
    
    # get model and model args
    model_type = vars(modules)[args.model] 
    parser = model_type.add_model_specific_args(parser)
    
    # get dataset and dataset args
    dataset_type = vars(datasets)[args.dataset]
    parser = dataset_type.add_dataset_specific_args(parser)

    args = parse_args(parser, argv)

    os.makedirs(args.exp_dir, exist_ok=True)
    os.makedirs(args.path_db, exist_ok=True)

    # save config
    with open(os.path.join(args.exp_dir, 'config.yaml'), 'w') as cfg_file:
        yaml.dump(args.__dict__, cfg_file)

    logger = TensorBoardLogger(args.exp_dir) #, name="my_model"
    
    datamodule = dataset_type(**args.__dict__)
    
    model = model_type(**args.__dict__)

    if args.resume_training:
        ckpts = list(filter(lambda x:'epoch=' in x, os.listdir(args.exp_dir)))
        latest_epoch = max([int( x.replace('epoch=','').replace('.ckpt','')) for x in ckpts])
        latest_ckpt = os.path.join(args.exp_dir, 'epoch=' + str(latest_epoch) + '.ckpt') 

        print('resuming from checkpoint', latest_ckpt)

        args.__dict__.update({'resume_from_checkpoint': latest_ckpt})

    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=args.exp_dir, save_top_k=3, mode='max', monitor='knn_acc', period=args.ckpt_period) # , filename='{epoch}-{knn_acc}' 

    trainer = pl.Trainer.from_argparse_args(args, logger=logger, checkpoint_callback=model_checkpoint, callbacks=[KNNEval(period=args.ckpt_period)])    
    trainer.fit(model, datamodule)

    best_ckpt = trainer.checkpoint_callback.best_model_path
    best_model = model_type.load_from_checkpoint(checkpoint_path=best_ckpt)
    pretrain_result = trainer.test(model=best_model, datamodule=datamodule)[0]

    lincls_results = lincls(args, best_model)
    
    print('test results')
    for k,v in lincls_results.items():
        print(k, v)

    df = pd.DataFrame()
    output_dict = {
        'exp_name': args.exp_name,
        'exp_dir': args.exp_dir,
        'model': args.model,
        'dataset': args.dataset,
    }

    output_dict.update(pretrain_result)
    output_dict.update(lincls_results)

    df = df.append(output_dict, ignore_index=True)
    df.to_csv(os.path.join(args.path_db, args.exp_name + '_db.csv'))







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

    val_features = []
    val_target = []
    with torch.no_grad():
        for batch, target in val_loader:
            input_, seq_len = batch 
            val_features.append(encoder(input_.cuda(), seq_len).cpu())
            val_target.append(torch.Tensor(target))
    
    val_features = torch.cat(val_features, 0)
    val_target = torch.cat(val_target, 0)
    
    batch_size = 512 if 'NTU' in args.dataset else 128

    datamodule = datasets.FeatureDataModule(train_features, train_target, val_features, val_target, num_workers=4, batch_size=batch_size)
    
    model_checkpoint = pl.callbacks.ModelCheckpoint(filepath=os.path.join(args.exp_dir,'lincls'), save_top_k=1, mode='max', monitor='val_acc1_agg', period=1) # , filename='{epoch}-{knn_acc}' 

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
    
    best_ckpt = trainer.checkpoint_callback.best_model_path
    best_model = modules.LinearClassifierMod.load_from_checkpoint(checkpoint_path=best_ckpt)
        
    result_dict = {}

    # lincls_result = trainer.test(model=best_model, verbose=True)[0]
    lincls_result = trainer.test(model=best_model, datamodule=datamodule)[0]

    result_dict.update(lincls_result)
    
    nn_ = 9 if 'NTU' in args.dataset else 1
    knn_acc = modules.knn(train_features.numpy(), val_features.numpy(), train_target.numpy(), val_target.numpy(), nn=nn_)
    result_dict['knn_acc'] = knn_acc

    return result_dict

if __name__ == '__main__':
    cli_main()

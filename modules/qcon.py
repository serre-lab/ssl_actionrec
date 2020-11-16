from argparse import ArgumentParser
from typing import Union
from warnings import warn

import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from pl_bolts.metrics import precision_at_k, mean

from models import Encoder, TemporalAveragePooling, LastHidden, LastSeqHidden, TileLast



class QCon(pl.LightningModule):

    def __init__(self,
                #  base_encoder: Union[str, torch.nn.Module] = 'resnet18',
                 emb_dim: int = 256,
                 dict_size: int = 65536,
                 vector_momentum: float = 0.999,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 0.03,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 optimizer: str = 'adam',
                #  datamodule: pl.LightningDataModule = None,
                #  batch_size: int = 256,
                 use_mlp: bool = False,
                 num_workers: int = 4,
                 epochs: int = 800,
                 lr_decay_rate: float = 0.1,
                 output_feature: str = 'last',
                 bidirectional: bool = False,
                 input_dim: int = 75,
                 input_embedding: int = 128,
                 hidden_dim: int = 512,
                 cell: str = 'LSTM',
                 num_layers: int = 1,
                 *args, **kwargs):
                 
        """
        Args:
            base_encoder: torchvision model name or torch.nn.Module
            emb_dim: feature dimension (default: 128)
            num_negatives: queue size; number of negative keys (default: 65536)
            encoder_momentum: moco momentum of updating key encoder (default: 0.999)
            softmax_temperature: softmax temperature (default: 0.07)
            learning_rate: the learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
            datamodule: the DataModule (train, val, test dataloaders)
            data_dir: the directory to store data
            batch_size: batch size
            use_mlp: add an mlp to the encoders
            num_workers: workers for the loaders
        """

        super().__init__()
        self.save_hyperparameters()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder = self.init_encoders()
        
        if use_mlp:  # hack: brute-force replacement
            self.trans = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, emb_dim))
            # nn.init.xavier_uniform_(param)

        else:
            self.trans = nn.Linear(hidden_dim, emb_dim)

        if output_feature=='tap':
            self.reduce_encoder_output = TemporalAveragePooling() # x [B,T,F]
        elif output_feature=='last':
            self.reduce_encoder_output = LastHidden() # x [B,T,F]
        elif output_feature=='seqlast':
            self.reduce_encoder_output = LastSeqHidden() # x [B,T,F]

        for param in list(self.encoder.parameters()):
            if param.dim() == 2:
                nn.init.xavier_uniform_(param)
                # nn.init.orthogonal_(param)

        # create the queue
        self.register_buffer("vec_dict", torch.randn(emb_dim, dict_size))
        self.vec_dict = nn.functional.normalize(self.vec_dict, dim=0)

        self.vec_dict = nn.Embedding(dict_size, emb_dim)
        # self.vec_dict.weight.data.uniform_(-1, 1)
        self.vec_dict.weight.data.normal_(-1, 1)
        # nn.init.orthogonal_(self.vec_dict.weight)


    def get_encoder(self):

        class ModelEncoder(nn.Module):
            def __init__(self, encoder, reducer):
                super().__init__()
                self.encoder = encoder
                self.reducer = reducer
            
            def forward(self, input_, seq_len):
                return self.reducer(self.encoder(input_), seq_len)
        return ModelEncoder(self.encoder, self.reduce_encoder_output)

    def init_encoders(self):
        """
        Override to add your own encoders
        """

        encoder = Encoder(input_dim=self.hparams.input_dim, 
                            hidden_dim=self.hparams.hidden_dim, 
                            bidirectional=self.hparams.bidirectional, 
                            embedding=self.hparams.input_embedding, 
                            cell=self.hparams.cell, 
                            num_layers=self.hparams.num_layers)

        return encoder

    def get_representations(self, input_, seq_len):
        return self.reduce_encoder_output(self.encoder(input_), seq_len=seq_len)

    def forward(self, input_1, input_2):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        # q = self.encoder_q_forward(input_q[0], input_q[1])  # queries: NxC
        
        vecs1 = self.trans(self.reduce_encoder_output(self.encoder(input_1[0]), input_1[1]))
        vecs2 = self.trans(self.reduce_encoder_output(self.encoder(input_2[0]), input_2[1]))
        vecs = nn.functional.normalize(torch.cat([vecs1,vecs2],0), dim=1)
        
        emb_vecs = nn.functional.normalize(self.vec_dict.weight, dim=1)

        # logs = torch.einsum('nc,ck->nk', [vecs, emb_vecs.t()]).unsqueeze(-1)
        logs = torch.einsum('nc,ck->nk', [vecs, emb_vecs.t()])
        logs /= self.hparams.softmax_temperature
        
        # probs_1 = nn.functional.softmax(logs[0::2])
        # probs_2 = nn.functional.softmax(logs[1::2])
        b = logs.shape[0]//2
        probs_1 = nn.functional.softmax(logs[:b], dim=1)
        probs_2 = nn.functional.softmax(logs[b:], dim=1)

        # cos_emb = torch.einsum('nc,ck->nk', [emb_vecs, emb_vecs.t()]) * (1 - torch.eye(self.hparams.dict_size, device=self.device))

        return probs_1, probs_2

    def shared_step(self, batch, batch_idx):
        
        # (input_1, input_2), labels = batch
        # shape1 = input_1[0].shape
        # shape2 = input_2[0].shape

        # pad_list = [0]*(len(shape1)*2) 
        # pad_list_1 = pad_list
        # pad_list_2 = pad_list
        # pad_list_1[3] = shape1[1] - max(shape1[1], shape2[1])
        # pad_list_2[3] = shape2[1] - max(shape1[1], shape2[1])

        # input_ = torch.cat([F.pad(input=input_1[0], pad=pad_list_1, mode='constant', value=0),
        #             F.pad(input=input_2[0], pad=pad_list_2, mode='constant', value=0)], 0)
        
        # input_ = (input_, input_1[1] + input_2[1]) 

        # probs_1, probs_2, cos_emb = self(input_ = input_)
        
        (input_1, input_2), labels = batch

        # probs_1, probs_2, cos_emb = self(input_1 = input_1, input_2 = input_2)
        probs_1, probs_2 = self(input_1 = input_1, input_2 = input_2)
        
        mask = torch.eye(probs_1.shape[0], device=self.device)

        # attractive
        # loss_att = - ((probs_2 * probs_1.log()).sum(1) + (probs_1 * probs_2.log()).sum(1))

        # attractive + repulsive
        h_12 = (probs_1[:,None,:] * probs_2[None,:,:].log()).sum(2)
        h_21 = (probs_2[:,None,:] * probs_1[None,:,:].log()).sum(2)
        
        loss_att = - ((h_12 * mask).mean() + (h_21 * mask).mean())
        loss_rep = ((h_12 * (1-mask)).mean() + (h_21 * (1-mask)).mean())

        acc1 = (probs_1.max(1)[1] == probs_2.max(1)[1]).sum().float() * 100 / probs_1.shape[0] 
        
        # print(probs_1.max(1)[1].float().var())
        # cos_emb = cos_emb.mean()
        # print(probs_1.max(1)[0].mean())

        # return loss_att, loss_rep, acc1, cos_emb
        return loss_att, loss_rep, acc1

    # def backward(self, trainer, loss, optimizer, optimizer_idx):
    #     loss.backward()
        
    def training_step(self, batch, batch_idx):

        loss_att, loss_rep, acc1 = self.shared_step(batch, batch_idx)

        emb_vecs = nn.functional.normalize(self.vec_dict.weight, dim=1)
        cos_emb = torch.einsum('nc,ck->nk', [emb_vecs, emb_vecs.t()]) * (1 - torch.eye(self.hparams.dict_size, device=self.device))
        cos_emb = cos_emb.mean()

        loss = self.hparams.loss_att_weight * loss_att + self.hparams.loss_rep_weight * loss_rep + self.hparams.loss_emb_weight * (1+cos_emb)

        # opt_m, opt_d = self.optimizers()
        # self.manual_backward(loss, opt_m)
        
        # opt_m.step()
        # opt_m.zero_grad()
        
        # opt_d.step()
        # opt_d.zero_grad()

        self.log('train_loss', loss)
        self.log('train_acc1', acc1)
        self.log('train_loss_att', loss_att)
        self.log('train_loss_rep', loss_rep)
        self.log('train_loss_emb', cos_emb)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        loss_att, loss_rep, acc1 = self.shared_step(batch, batch_idx)

        results = {
            'val_loss_att': loss_att,
            'val_loss_rep': loss_rep,
            'val_acc1': acc1,
        }

        return results

    def validation_epoch_end(self, outputs):
        loss_att = mean(outputs, 'val_loss_att')
        loss_rep = mean(outputs, 'val_loss_rep')
        acc1 = mean(outputs, 'val_acc1')

        self.log('val_loss_att_agg', loss_att)
        self.log('val_loss_rep_agg', loss_rep)
        self.log('val_acc1_agg', acc1)


    def test_step(self, batch, batch_idx):

        loss_att, loss_rep, acc1 = self.shared_step(batch, batch_idx)

        results = {
            'test_loss_att': loss_att.cpu(),
            'test_loss_rep': loss_rep.cpu(),
            'test_acc1': acc1.cpu(),
        }

        return results

    def test_epoch_end(self, outputs):
        test_loss_att = mean(outputs, 'test_loss_att')
        test_loss_rep = mean(outputs, 'test_loss_rep')
        test_acc1 = mean(outputs, 'test_acc1')

        results = {
            'test_loss_att': test_loss_att.item(),
            'test_loss_rep': test_loss_rep.item(),
            'test_con_acc1': test_acc1.item(),
        }

        return results


    def configure_optimizers(self):
        if self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), self.hparams.learning_rate,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)

            # eta_min = self.hparams.learning_rate * (self.hparams.lr_decay_rate ** 3) * 0.1
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.epochs, eta_min, -1)
            
            lamda_fun = lambda epoch: self.hparams.lr_decay_rate ** np.sum(epoch > np.asarray([30,70]))            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lamda_fun)
            return [optimizer], [scheduler]      
        # elif self.hparams.optimizer == 'adam':
        else:
            # optimizer = torch.optim.Adam(self.parameters(), self.hparams.learning_rate)
            optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.trans.parameters()), self.hparams.learning_rate)
            # optimizer1 = torch.optim.Adam(list(self.encoder.parameters()) + list(self.trans.parameters()), self.hparams.learning_rate)
            # optimizer2 = torch.optim.SGD(self.vec_dict.parameters(), 1 - self.hparams.vector_momentum)
            
            return optimizer
            # return [optimizer1, optimizer2]

    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--base_encoder', type=str, default='resnet18')
        parser.add_argument('--emb_dim', type=int, default=256)
        parser.add_argument('--dict_size', type=int, default=4096)
        parser.add_argument('--vector_momentum', type=float, default=0.999)
        parser.add_argument('--softmax_temperature', type=float, default=0.07)
        parser.add_argument('--learning_rate', type=float, default=0.03)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument('--output_feature', type=str, default='last') #tap 
        
        parser.add_argument('--epochs', type=int, default=800)
        parser.add_argument('--lr_decay_rate', type=float, default=0.1) 
        parser.add_argument('--optimizer', type=str, default='adam') 

        parser.add_argument('--loss_att_weight', type=float, default=1) 
        parser.add_argument('--loss_rep_weight', type=float, default=1) 
        parser.add_argument('--loss_emb_weight', type=float, default=0) 
        
        # model parameters
        parser.add_argument('--bidirectional', type=bool, default=False) 
        parser.add_argument('--input_dim', type=int, default=75)
        parser.add_argument('--input_embedding', type=str, default=None)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--cell', type=str, default='LSTM')
        parser.add_argument('--num_layers', type=int, default=1)

        # use_decoder
        
        return parser

def get_shuffle_ids(bsz):
    """generate shuffle ids for ShuffleBN"""
    forward_inds = torch.randperm(bsz).long().cuda()
    backward_inds = torch.zeros(bsz).long().cuda()
    value = torch.arange(bsz).long().cuda()
    backward_inds.index_copy_(0, forward_inds, value)
    return forward_inds, backward_inds


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

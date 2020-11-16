from argparse import ArgumentParser
from typing import Union
from warnings import warn

import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

# from pl_bolts.metrics import precision_at_k, mean
from pl_bolts.metrics import mean

from models import Encoder, Decoder, TemporalAveragePooling, LastHidden, TileLast, LastSeqHidden, TileSeqLast, Tile

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# changes: init, init_encoders, _dequeue_and_enqueue, forward, training_step, validation_step, validation_epoch_end
# BU/BUTD

# BUQ
# BUQTD
# BUQTDQ

def precision_at_k(output, target, top_k=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(top_k)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in top_k:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class TDCon(pl.LightningModule):

    def __init__(self,
                 num_negatives: int = 512,
                 maximum_timesteps: int = 50,
                 encoder_momentum: float = 0.999,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 0.03,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 optimizer: str = 'adam',
                 decoder_input: str = 'seqlast',
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

        super().__init__()
        self.save_hyperparameters()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k, self.decoder_q, self.decoder_k = self.init_encoders()
        
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # for param_q, param_k in zip(self.trans_q.parameters(), self.trans_k.parameters()):
        #     param_k.data.copy_(param_q.data)  # initialize
        #     param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.decoder_q.parameters(), self.decoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if output_feature=='tap':
            self.reduce_encoder_output = TemporalAveragePooling() # x [B,T,F]
        elif output_feature=='last':
            self.reduce_encoder_output = LastHidden() # x [B,T,F]
        elif output_feature=='seqlast':
            self.reduce_encoder_output = LastSeqHidden() # x [B,T,F]

        if decoder_input=='last':
            self.decoder_input = TileLast()
        elif decoder_input=='seqlast':
            self.decoder_input = TileSeqLast()
        elif decoder_input=='all':
            self.decoder_input = nn.Identity()

        self.criterion = nn.L1Loss(reduction="none")
        
        self.register_buffer("queue", torch.randn(num_negatives, maximum_timesteps, input_dim))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.k_modules = nn.ModuleList([self.encoder_k, self.decoder_k])
        self.q_modules = nn.ModuleList([self.encoder_q, self.decoder_q])

    def get_encoder(self):
        class ModelEncoder(nn.Module):
            def __init__(self, encoder, reducer):
                super().__init__()
                self.encoder = encoder
                self.reducer = reducer
            
            def forward(self, input_, seq_len):
                return self.reducer(self.encoder(input_), seq_len)
        return ModelEncoder(self.encoder_q, self.reduce_encoder_output)
        # return nn.Sequential(self.encoder, self.reduce_encoder_output)
        
    def init_encoders(self):
        """
        Override to add your own encoders
        """

        encoder_q = Encoder(input_dim=self.hparams.input_dim, 
                            hidden_dim=self.hparams.hidden_dim, 
                            bidirectional=self.hparams.bidirectional, 
                            embedding=self.hparams.input_embedding, 
                            cell=self.hparams.cell, 
                            num_layers=self.hparams.num_layers)

        encoder_k = Encoder(input_dim=self.hparams.input_dim, 
                            hidden_dim=self.hparams.hidden_dim, 
                            bidirectional=self.hparams.bidirectional, 
                            embedding=self.hparams.input_embedding, 
                            cell=self.hparams.cell, 
                            num_layers=self.hparams.num_layers)

        decoder_q = Decoder(input_dim=self.hparams.hidden_dim, 
                            hidden_dim=self.hparams.hidden_dim, 
                            output_dim=self.hparams.input_dim,
                            bidirectional=self.hparams.bidirectional, 
                            cell=self.hparams.cell, 
                            num_layers=self.hparams.num_layers)

        decoder_k = Decoder(input_dim=self.hparams.hidden_dim, 
                            hidden_dim=self.hparams.hidden_dim, 
                            output_dim=self.hparams.input_dim,
                            bidirectional=self.hparams.bidirectional, 
                            cell=self.hparams.cell, 
                            num_layers=self.hparams.num_layers)
        
        return encoder_q, encoder_k, decoder_q, decoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """

        for param_q, param_k in zip(self.q_modules.parameters(), self.k_modules.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)
    
    def get_representations(self, input_, seq_len):
        
        return self.reduce_encoder_output(self.encoder_q(input_), seq_len=seq_len)

    def get_q_outputs(self, input_q, seq_len, out_len):

        feats_q = self.encoder_q(input_q)  # queries: NxC
        rec_q = self.decoder_q(self.decoder_input(feats_q, seq_len=seq_len, out_len=out_len))

        return rec_q

    def get_k_outputs(self, input_k, seq_len, out_len):
        
        feats_k = self.encoder_k(input_k)  # queries: NxC
        rec_k = self.decoder_k(self.decoder_input(feats_k, seq_len=seq_len, out_len=out_len))

        return rec_k

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        # if self.use_ddp or self.use_ddp2:
        #     keys = concat_all_gather(keys)
        #     keys_td = concat_all_gather(keys_td)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity
        # keys = keys.reshape([batch_size, -1])
        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + batch_size) > self.hparams.num_negatives:
            self.queue[ptr:ptr + batch_size] = keys[:self.hparams.num_negatives - ptr]
            self.queue[:ptr + batch_size - self.hparams.num_negatives] = keys[self.hparams.num_negatives - ptr:]
        else:
            self.queue[ptr:ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[0] = ptr

    def forward(self, input_q, input_k, ground_truth):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        out_len = ground_truth[0].shape[1]
        # compute query features
        rec_q = self.get_q_outputs(input_q[0], seq_len=input_q[1], out_len=out_len)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            rec_k = self.get_k_outputs(input_k[0], seq_len=input_k[1], out_len=self.hparams.maximum_timesteps)

        err_pos = masked_loss(self.criterion, rec_q, rec_k, ground_truth[1], reduction='none')
        err_neg = masked_loss(self.criterion, rec_q[:,None], self.queue[None,:], ground_truth[1], reduction='none')
        
        # neg_mse = ((rec_q[:,:,:,None] - self.queue.clone().detach()[None,:,:,:])**2).mean([1,2])
        # pos_mse = torch.stack([((rec_q - ground_truth)**2).mean([1,2]), ((rec_q - rec_k)**2).mean([1,2])], 1)

        logits_pos = - err_pos / self.hparams.softmax_temperature
        logits_neg = - err_neg / self.hparams.softmax_temperature

        logits = torch.cat([logits_pos[:,None], logits_neg], dim=1)
        
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        return rec_q, rec_k, logits, labels
        
    def shared_step(self, batch, batch_idx):

        (input_1, input_2, ground_truth), _ = batch
        
        gt = ground_truth[0].view(list(ground_truth[0].shape[:2]) + [-1])
        rec_q, rec_k, logits, target = self(input_1, input_2, ground_truth)
        loss_td = F.cross_entropy(logits.float(), target.long())
        loss_gt = masked_loss(self.criterion, rec_q, gt, ground_truth[1])

        acc1, acc5 = precision_at_k(logits, target, top_k=(1, 5))

        return rec_k, loss_gt, loss_td, acc1, acc5

    def training_step(self, batch, batch_idx):

        rec_k, loss_gt, loss_td, acc1, acc5 = self.shared_step(batch, batch_idx)

        loss = self.hparams.con_weight * loss_td + self.hparams.rec_weight * loss_gt
        
        self._dequeue_and_enqueue(rec_k)
        
        self.log('train_loss_rec', loss_gt)
        self.log('train_loss_con', loss_td)
        self.log('train_acc1', acc1)
        self.log('train_acc5', acc5)
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        _, loss_gt, loss_td, acc1, acc5 = self.shared_step(batch, batch_idx)
        
        results = {
            'val_loss_rec': loss_gt,
            'val_loss_con': loss_td,
            'val_acc1': acc1,
            'val_acc5': acc5,
        }
        return results    

    def validation_epoch_end(self, outputs):
        
        val_loss_rec = mean(outputs, 'val_loss_rec')
        val_loss_con = mean(outputs, 'val_loss_con')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')
        self.log('val_loss_rec_agg', val_loss_rec)
        self.log('val_loss_con_agg', val_loss_con)
        self.log('val_acc1_agg', val_acc1)
        self.log('val_acc5_agg', val_acc5)

    def test_step(self, batch, batch_idx):
        _, loss_gt, loss_td, acc1, acc5 = self.shared_step(batch, batch_idx)       
        results = {
            'test_loss_rec': loss_gt.cpu(),
            'test_loss_con': loss_td.cpu(),
            'test_acc1': acc1.cpu(),
            'test_acc5': acc5.cpu(),
        }
        return results

    def test_epoch_end(self, outputs):
        test_loss_rec = mean(outputs, 'test_loss_rec')
        test_loss_con = mean(outputs, 'test_loss_con')
        test_acc1 = mean(outputs, 'test_acc1')
        test_acc5 = mean(outputs, 'test_acc5')
        results = {
            'test_loss_rec': test_loss_rec.item(),
            'test_loss_con': test_loss_con.item(),
            'test_acc1': test_acc1.item(),
            'test_acc5': test_acc5.item(),
        }
        return results


    def configure_optimizers(self):
        optimizer = torch.optim.Adam( 
            filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate
        )
        return optimizer
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        parser.add_argument('--decoder_input', type=str, default='seqlast') #all 
        parser.add_argument('--output_feature', type=str, default='seqlast') #tap 

        parser.add_argument('--num_negatives', type=int, default=512)
        parser.add_argument('--maximum_timesteps', type=int, default=50)
        parser.add_argument('--encoder_momentum', type=float, default=0.999)
        parser.add_argument('--softmax_temperature', type=float, default=0.07)
        parser.add_argument('--learning_rate', type=float, default=0.03)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        
        parser.add_argument('--epochs', type=int, default=800)
        parser.add_argument('--lr_decay_rate', type=float, default=0.1) 
        parser.add_argument('--optimizer', type=str, default='adam') 
        parser.add_argument('--con_weight', type=float, default=1) 
        parser.add_argument('--rec_weight', type=float, default=1) 
        
        
        # model parameters
        parser.add_argument('--bidirectional', type=bool, default=False) 
        parser.add_argument('--input_dim', type=int, default=75)
        parser.add_argument('--input_embedding', type=str, default=None)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--cell', type=str, default='LSTM')
        parser.add_argument('--num_layers', type=int, default=1)

        # use_decoder
        
        return parser


def masked_loss(criterion, pred, target, seq_len, reduction='mean', pad=True):
    
    if pad:
        pred = pad_to_size(pred, target.shape[-2] - pred.shape[-2], dim=-2)
    
    target = target.reshape(list(target.shape[:-2]) + list(pred.shape[-2:]))
    mask = torch.arange(pred.shape[-2]).long().expand(pred.shape[:-1])

    mask = mask < torch.Tensor(seq_len).long().view([-1] + [1]*(len(mask.shape)-1) )
    
    mask = mask.to(pred.device)
    
    pred = pred - target
    loss = criterion(pred, torch.zeros_like(pred))
    loss = loss.mean(-1)
    loss = (loss*mask).sum(-1)/mask.sum(-1)
    
    if reduction=='mean':
        loss = loss.mean()
    
    return loss

def pad_to_size(t, size, dim):
    
    pad_list = [0]*2*len(t.shape)
    dim = (len(t.shape) - dim -1)%len(t.shape)
    pad_list[dim*2+1] = size
    return F.pad(t,pad_list,'constant')
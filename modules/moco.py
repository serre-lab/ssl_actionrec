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



class MocoV2(pl.LightningModule):

    def __init__(self,
                #  base_encoder: Union[str, torch.nn.Module] = 'resnet18',
                 emb_dim: int = 128,
                 num_negatives: int = 65536,
                 encoder_momentum: float = 0.999,
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
        self.encoder_q, self.encoder_k = self.init_encoders()
        
        if use_mlp:  # hack: brute-force replacement
            # dim_mlp = self.encoder_q.output_dim
            self.trans_k = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, emb_dim))
            self.trans_q = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, emb_dim))
        else:
            self.trans_k = nn.Linear(hidden_dim, emb_dim)
            self.trans_q = nn.Linear(hidden_dim, emb_dim)

        if output_feature=='tap':
            self.reduce_encoder_output = TemporalAveragePooling() # x [B,T,F]
        elif output_feature=='last':
            self.reduce_encoder_output = LastHidden() # x [B,T,F]
        elif output_feature=='seqlast':
            self.reduce_encoder_output = LastSeqHidden() # x [B,T,F]

        for param in list(self.encoder_q.parameters()):
            if param.dim() == 2:
                nn.init.xavier_uniform_(param)
                # nn.init.orthogonal_(param)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.trans_q.parameters(), self.trans_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # self.encoder_k_forward = nn.Sequential(self.encoder_k, self.reduce_encoder_output, self.trans_k)
        # self.encoder_q_forward = nn.Sequential(self.encoder_q, self.reduce_encoder_output, self.trans_q)
        
        self.encoder_k_module = nn.ModuleList([self.encoder_k, self.reduce_encoder_output, self.trans_k])
        self.encoder_q_module = nn.ModuleList([self.encoder_q, self.reduce_encoder_output, self.trans_q])

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))



    def get_encoder(self):

        class ModelEncoder(nn.Module):
            def __init__(self, encoder, reducer):
                super().__init__()
                self.encoder = encoder
                self.reducer = reducer
            
            def forward(self, input_, seq_len):
                return self.reducer(self.encoder(input_), seq_len)
        return ModelEncoder(self.encoder_q, self.reduce_encoder_output)

        # return nn.Sequential(self.encoder_q, self.reduce_encoder_output)
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

        return encoder_q, encoder_k

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """

        # for param_q, param_k in zip(self.encoder_q_forward.parameters(), self.encoder_k_forward.parameters()):
        #     em = self.hparams.encoder_momentum
        #     param_k.data = param_k.data * em + param_q.data * (1. - em)
            
        for param_q, param_k in zip(self.encoder_q_module.parameters(), self.encoder_k_module.parameters()):
            em = self.hparams.encoder_momentum
            param_k.data = param_k.data * em + param_q.data * (1. - em)
            

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.use_ddp or self.use_ddp2:
            keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + batch_size) > self.hparams.num_negatives:
            self.queue[:, ptr:ptr + batch_size] = keys.T[:self.hparams.num_negatives - ptr]
            self.queue[:, :ptr + batch_size - self.hparams.num_negatives] = keys.T[self.hparams.num_negatives - ptr:]
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[0] = ptr

    def get_representations(self, input_, seq_len):
        return self.reduce_encoder_output(self.encoder_q(input_), seq_len=seq_len)

    def forward(self, input_q, input_k):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        # q = self.encoder_q_forward(input_q[0], input_q[1])  # queries: NxC
        
        q = self.trans_q(self.reduce_encoder_output(self.encoder_q(input_q[0]), input_q[1]))
        q = nn.functional.normalize(q, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            
            # shuffle_ids, reverse_ids = get_shuffle_ids(input_k[0].shape[0])
            
            # k = self.trans_k(self.reduce_encoder_output(self.encoder_k(input_k[0][shuffle_ids]), torch.Tensor(input_k[1]).long()[shuffle_ids]))
            # k = nn.functional.normalize(k, dim=1)
            # k = k[reverse_ids]

            k = self.trans_k(self.reduce_encoder_output(self.encoder_k(input_k[0]), input_k[1]))
            k = nn.functional.normalize(k, dim=1)

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        return k, logits, labels

    def shared_step(self, batch, batch_idx):
        # (input_1, _, input_2, _), _ = batch
        # ((input_1, seq_len_1), (input_2, seq_len_2)), labels = batch
        
        (input_1, input_2), labels = batch

        k, output, target = self(input_q=input_1, input_k=input_2)
        loss = F.cross_entropy(output.float(), target.long())
        
        # dequeue and enqueue
        # self._dequeue_and_enqueue(k)

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        return loss, acc1, acc5, k

    def training_step(self, batch, batch_idx):

        loss, acc1, acc5, k = self.shared_step(batch, batch_idx)

        self._dequeue_and_enqueue(k)

        self.log('train_loss', loss)
        self.log('train_acc1', acc1)
        self.log('train_acc5', acc5)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        loss, acc1, acc5, _ = self.shared_step(batch, batch_idx)

        results = {
            'val_loss': loss,
            'val_acc1': acc1,
            'val_acc5': acc5
        }

        return results

    def validation_epoch_end(self, outputs):
        val_loss = mean(outputs, 'val_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')

        self.log('val_loss_agg', val_loss)
        self.log('val_acc1_agg', val_acc1)
        self.log('val_acc5_agg', val_acc5)


    def test_step(self, batch, batch_idx):

        loss, acc1, acc5, _ = self.shared_step(batch, batch_idx)

        results = {
            'test_loss': loss.cpu(),
            'test_acc1': acc1.cpu(),
            'test_acc5': acc5.cpu(),
        }

        return results

    def test_epoch_end(self, outputs):
        test_loss = mean(outputs, 'test_loss')
        test_acc1 = mean(outputs, 'test_acc1')
        test_acc5 = mean(outputs, 'test_acc5')

        results = {
            'test_con_loss': test_loss.item(),
            'test_con_acc1': test_acc1.item(),
            'test_con_acc5': test_acc5.item(),
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
            optimizer = torch.optim.Adam(self.parameters(), self.hparams.learning_rate)
            return optimizer

    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--base_encoder', type=str, default='resnet18')
        parser.add_argument('--emb_dim', type=int, default=128)
        parser.add_argument('--num_negatives', type=int, default=65536)
        parser.add_argument('--encoder_momentum', type=float, default=0.999)
        parser.add_argument('--softmax_temperature', type=float, default=0.07)
        parser.add_argument('--learning_rate', type=float, default=0.03)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument('--output_feature', type=str, default='last') #tap 
        
        parser.add_argument('--epochs', type=int, default=800)
        parser.add_argument('--lr_decay_rate', type=float, default=0.1) 
        parser.add_argument('--optimizer', type=str, default='adam') 

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

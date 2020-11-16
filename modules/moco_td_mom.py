from argparse import ArgumentParser
from typing import Union
from warnings import warn

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from pl_bolts.metrics import precision_at_k, mean

from models import Encoder, Decoder, TemporalAveragePooling, LastHidden, TileLast, LastSeqHidden, TileSeqLast

# changes: init, init_encoders, _dequeue_and_enqueue, forward, training_step, validation_step, validation_epoch_end
# BU/BUTD

# BUQ
# BUQTD
# BUQTDQ

class MocoV2TDMom(pl.LightningModule):

    def __init__(self,
                 emb_dim: int = 128,
                 num_negatives: int = 65536,
                 num_td_negatives: int = 512,
                 encoder_momentum: float = 0.999,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 0.03,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 decoder_input: str = 'last',
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

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.trans_q.parameters(), self.trans_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        for param_q, param_k in zip(self.decoder_q.parameters(), self.decoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if decoder_input=='last':
            self.decoder_input = TileLast()
        elif decoder_input=='all':
            self.decoder_input = nn.Identity()

        # create the queue
        self.register_buffer("queue", torch.randn(emb_dim, num_negatives))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.k_modules = nn.ModuleList([self.encoder_k, self.trans_k, self.decoder_k])
        self.q_modules = nn.ModuleList([self.encoder_q, self.trans_q, self.decoder_q])

    def get_encoder(self):
        return nn.Sequential(self.encoder_q, self.reduce_encoder_output)
        
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

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        if self.use_ddp or self.use_ddp2:
            keys = concat_all_gather(keys)
            keys_td = concat_all_gather(keys_td)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[0] = ptr
        
    def get_q_outputs(self, input_q):

        feats_q = self.encoder_q(input_q)  # queries: NxC
        q = self.trans_q(self.reduce_encoder_output(feats_q))
        q = nn.functional.normalize(q, dim=1)
        rec_q = self.decoder_q(self.decoder_input(feats_q))

        return q, rec_q

    def get_k_outputs(self, input_k):
        
        feats_k = self.encoder_k(input_k)  # queries: NxC
        k = self.trans_k(self.reduce_encoder_output(feats_k))
        k = nn.functional.normalize(k, dim=1)
        rec_k = self.decoder_k(self.decoder_input(feats_k))

        return k, rec_k

    def forward(self, input_q, input_k, ground_truth):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q, rec_q = self.get_q_outputs(input_q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k, rec_k = self.get_k_outputs(input_k)

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
        
        return k, logits, labels, rec_k, rec_q

    def training_step(self, batch, batch_idx):

        # (input_1, _, input_2, _), _ = batch
        (input_1, input_2, ground_truth), _ = batch

        k, output, target, rec_k, rec_q = self(input_q=input_1, input_k=input_2)
        nll_bu = F.cross_entropy(output.float(), target.long())
        
        loss_qgt = ((rec_q - ground_truth)**2).mean()
        loss_qk = ((rec_q - rec_k)**2).mean()
        
        loss_td = loss_qgt + loss_qk

        loss = nll_bu + loss_td

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        self.log('train_mse_k', loss_qk)
        self.log('train_mse_gt', loss_qgt)
        self.log('train_bu_loss', nll_bu)
        self.log('train_td_loss', loss_td)
        self.log('train_acc1', acc1)
        self.log('train_acc5', acc5)

        return {'loss': loss} #, 'log': log, 'progress_bar': log

    def validation_step(self, batch, batch_idx):

        (input_1, input_2, ground_truth), _ = batch

        k, output, target, rec_k, nll_td, pos_mse= self(input_q=input_1, input_k=input_2)
        nll_bu = F.cross_entropy(output.float(), target.long())
        # loss = nll_bu + nll_td
        
        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {
            'val_mse_k': pos_mse[:,1].mean(),
            'val_mse_gt': pos_mse[:,0].mean(),
            'val_bu_loss': nll_bu,
            'val_td_loss': nll_td,
            'val_acc1': acc1,
            'val_acc5': acc5
        }

        loss_qgt = ((rec_q - ground_truth)**2).mean()
        loss_qk = ((rec_q - rec_k)**2).mean()
        
        loss_td = loss_qgt + loss_qk

        # loss = nll_bu + loss_td
        
        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {
            'val_mse_k': loss_qk,
            'val_mse_gt': loss_qgt,
            'val_bu_loss': nll_bu,
            'val_td_loss': loss_td,
            'val_acc1': acc1,
            'val_acc5': acc5
        }

        return results

    def validation_epoch_end(self, outputs):
        val_mse_k = mean(outputs, 'val_mse_k')
        val_mse_gt = mean(outputs, 'val_mse_gt')
        val_bu_loss = mean(outputs, 'val_bu_loss')
        val_td_loss = mean(outputs, 'val_td_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')
        
        self.log('val_mse_k_agg',   val_mse_k)
        self.log('val_mse_gt_agg',  val_mse_gt)
        self.log('val_bu_loss_agg', val_bu_loss)
        self.log('val_td_loss_agg', val_td_loss)
        self.log('val_acc1_agg',    val_acc1)
        self.log('val_acc5_agg',    val_acc5)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), self.hparams.learning_rate,
                                    momentum=self.hparams.momentum,
                                    weight_decay=self.hparams.weight_decay)
        return optimizer
        # eta_min = self.hparams.learning_rate * (self.hparams.lr_decay_rate ** 3) * 0.1
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.epochs, eta_min, -1)
        # return optimizer, scheduler
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # parser.add_argument('--base_encoder', type=str, default='resnet18')
        parser.add_argument('--emb_dim', type=int, default=128)
        parser.add_argument('--num_negatives', type=int, default=65536)
        parser.add_argument('--num_td_negatives', type=int, default=65536)
        parser.add_argument('--encoder_momentum', type=float, default=0.999)
        parser.add_argument('--softmax_temperature', type=float, default=0.07)
        parser.add_argument('--learning_rate', type=float, default=0.03)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument('--output_feature', type=str, default='last') #tap 
        parser.add_argument('--decoder_input', type=str, default='last') #all 

        parser.add_argument('--epochs', type=int, default=800)
        parser.add_argument('--lr_decay_rate', type=float, default=0.1) 

        # model parameters
        parser.add_argument('--bidirectional', type=bool, default=False) 
        parser.add_argument('--input_dim', type=int, default=75)
        parser.add_argument('--input_embedding', type=str, default=None)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--cell', type=str, default='LSTM')
        parser.add_argument('--num_layers', type=int, default=1)

        return parser




class MocoV2TDMomQ(MocoV2TDMom):

    def __init__(self,
                 emb_dim: int = 128,
                 num_negatives: int = 65536,
                 num_td_negatives: int = 512,
                 encoder_momentum: float = 0.999,
                 softmax_temperature: float = 0.07,
                 learning_rate: float = 0.03,
                 momentum: float = 0.9,
                 weight_decay: float = 1e-4,
                 decoder_input: str = 'last',
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

        super(MocoV2TDMomQ, self).__init__(
                        emb_dim = emb_dim,
                        num_negatives = num_negatives,
                        num_td_negatives = num_td_negatives,
                        encoder_momentum = encoder_momentum,
                        softmax_temperature = softmax_temperature,
                        learning_rate = learning_rate,
                        momentum = momentum,
                        weight_decay = weight_decay,
                        decoder_input = decoder_input,
                        use_mlp = use_mlp,
                        num_workers = num_workers,
                        epochs = epochs,
                        lr_decay_rate = lr_decay_rate,
                        output_feature = output_feature,
                        bidirectional = bidirectional,
                        input_dim = input_dim,
                        input_embedding = input_embedding,
                        hidden_dim = hidden_dim,
                        cell = cell,
                        num_layers = num_layers,)

        self.register_buffer("queue_td", torch.randn(300, input_dim, num_td_negatives))
        # self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_td_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, keys_td):
        # gather keys before updating queue
        if self.use_ddp or self.use_ddp2:
            keys = concat_all_gather(keys)
            keys_td = concat_all_gather(keys_td)

        batch_size = keys.shape[0]

        ##### bu queue

        ptr = int(self.queue_ptr)
        assert self.hparams.num_negatives % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_ptr[0] = ptr

        ##### td queue
        
        ptr = int(self.queue_td_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, :, ptr:ptr + batch_size] = keys_td.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        self.queue_td_ptr[0] = ptr

    def forward(self, input_q, input_k, ground_truth):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q, rec_q = self.get_q_outputs(input_q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k, rec_k = self.get_k_outputs(input_k)

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
        
        neg_mse = ((rec_q[:,:,:,None] - self.queue_td.clone().detach()[None,:,:,:])**2).mean([1,2])
        pos_mse = torch.stack([((rec_q - ground_truth)**2).mean([1,2]), ((rec_q - rec_k)**2).mean([1,2])], 1)

        logits_neg_mse = - neg_mse / self.hparams.softmax_temperature
        logits_pos_mse = - pos_mse / self.hparams.softmax_temperature

        nll_td = - torch.log(torch.exp(logits_pos_mse).sum(1)/ torch.exp(torch.cat([logits_pos_mse, logits_neg_mse], 1)).sum(1))
        nll_td = nll_td.mean()

        return k, logits, labels, rec_k, nll_td, pos_mse

    def training_step(self, batch, batch_idx):

        # (input_1, _, input_2, _), _ = batch
        (input_1, input_2, ground_truth), _ = batch

        k, output, target, rec_k, nll_td, pos_mse= self(input_q=input_1, input_k=input_2)
        nll_bu = F.cross_entropy(output.float(), target.long())
        loss = nll_bu + nll_td
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, rec_k)

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        self.log('train_mse_k', pos_mse[:,1].mean())
        self.log('train_mse_gt', pos_mse[:,0].mean())
        self.log('train_bu_loss', nll_bu)
        self.log('train_td_loss', nll_td)
        self.log('train_acc1', acc1)
        self.log('train_acc5', acc5)

        return {'loss': loss} #, 'log': log, 'progress_bar': log

    def validation_step(self, batch, batch_idx):

        (input_1, input_2, ground_truth), _ = batch

        k, output, target, rec_k, nll_td, pos_mse= self(input_q=input_1, input_k=input_2)
        nll_bu = F.cross_entropy(output.float(), target.long())
        # loss = nll_bu + nll_td
        
        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        results = {
            'val_mse_k': pos_mse[:,1].mean(),
            'val_mse_gt': pos_mse[:,0].mean(),
            'val_bu_loss': nll_bu,
            'val_td_loss': nll_td,
            'val_acc1': acc1,
            'val_acc5': acc5
        }

        return results

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


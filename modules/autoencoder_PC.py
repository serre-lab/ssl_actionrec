from argparse import ArgumentParser
from typing import Union
from warnings import warn

import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from pl_bolts.metrics import precision_at_k, mean

from models import Encoder, Decoder, TemporalAveragePooling, LastHidden, TileLast, LastSeqHidden, TileSeqLast, Tile

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# changes: init, init_encoders, _dequeue_and_enqueue, forward, training_step, validation_step, validation_epoch_end
# BU/BUTD

# BUQ
# BUQTD
# BUQTDQ

class AutoEncPC(pl.LightningModule):

    def __init__(self,
                 learning_rate: float = 0.03,
                 decoder_input: str = 'seqlast',
                 num_workers: int = 4,
                 epochs: int = 200,
                 output_feature: str = 'seqlast',
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
        self.encoder, self.decoder = self.init_encoders()
        
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

    def get_encoder(self):
        class ModelEncoder(nn.Module):
            def __init__(self, encoder, reducer):
                super().__init__()
                self.encoder = encoder
                self.reducer = reducer
            
            def forward(self, input_, seq_len):
                return self.reducer(self.encoder(input_), seq_len)
        return ModelEncoder(self.encoder, self.reduce_encoder_output)
        # return nn.Sequential(self.encoder, self.reduce_encoder_output)
        
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

        decoder = Decoder(input_dim=self.hparams.hidden_dim, 
                            hidden_dim=self.hparams.hidden_dim, 
                            output_dim=self.hparams.input_dim,
                            bidirectional=self.hparams.bidirectional, 
                            cell=self.hparams.cell, 
                            num_layers=self.hparams.num_layers)
        
        for param in list(encoder.parameters()) + list(decoder.parameters()):
            if param.dim() == 2:
                nn.init.xavier_uniform_(param)

        return encoder, decoder
    
    def get_representations(self, input_, seq_len):
        
        return self.reduce_encoder_output(self.encoder(input_), seq_len=seq_len)

    def forward(self, input_, seq_len, out_len):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        feat = self.encoder(input_) 
        feats = self.reduce_encoder_output(feat, seq_len=seq_len)
        rec = self.decoder(self.decoder_input(feat, seq_len=seq_len, out_len=out_len))

        return rec, feats

    def shared_step(self, batch, batch_idx):
        (input_, seq_len), _ = batch
        
        rec, feats = self(input_=input_, seq_len=seq_len, out_len=input_.shape[1])

        loss = masked_loss(self.criterion, rec, input_, seq_len)
        # loss = ((rec - ground_truth)**2).mean()
        return loss, feats

    def training_step(self, batch, batch_idx):

        loss, feats = self.shared_step(batch, batch_idx)
        
        self.log('train_loss', loss)
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        loss, feats = self.shared_step(batch, batch_idx)
        
        results = {
            'val_loss': loss,
        }

        return results    

    def validation_epoch_end(self, outputs):
        # print('val', outputs)
        
        val_loss = mean(outputs, 'val_loss')

        self.log('val_loss_agg', val_loss)

    def test_step(self, batch, batch_idx):

        loss, _ = self.shared_step(batch, batch_idx)
                
        results = {
            'test_loss': loss.cpu(),
        }

        return results

    def test_epoch_end(self, outputs):
        test_loss = mean(outputs, 'test_loss')
        
        results = {
            'test_rec_loss': test_loss.item(),
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
        parser.add_argument('--learning_rate', type=float, default=0.03)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--output_feature', type=str, default='last') #tap 
        parser.add_argument('--decoder_input', type=str, default='last') #all 

        # model parameters
        parser.add_argument('--bidirectional', type=bool, default=False) 
        parser.add_argument('--input_dim', type=int, default=75)
        parser.add_argument('--input_embedding', type=str, default=None)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--cell', type=str, default='LSTM')
        parser.add_argument('--num_layers', type=int, default=1)

        # use_decoder
        
        return parser





class VAEPC(pl.LightningModule):

    def __init__(self,
                learning_rate: float = 0.03,
                # decoder_input: str = 'seqlast',
                num_workers: int = 4,
                epochs: int = 200,
                output_feature: str = 'seqlast',
                bidirectional: bool = False,
                input_dim: int = 75,
                input_embedding: int = 128,
                hidden_dim: int = 512,
                cell: str = 'LSTM',
                num_layers: int = 1,
                latent_dim: int = 256,
                beta: float = 1,
                *args, **kwargs):

        super().__init__()
        self.save_hyperparameters()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder, self.decoder = self.init_encoders()
        
        if output_feature=='tap':
            self.reduce_encoder_output = TemporalAveragePooling() # x [B,T,F]
        elif output_feature=='last':
            self.reduce_encoder_output = LastHidden() # x [B,T,F]
        elif output_feature=='seqlast':
            self.reduce_encoder_output = LastSeqHidden() # x [B,T,F]

        # if decoder_input=='last':
        #     self.decoder_input = TileLast()
        # elif decoder_input=='seqlast':
        #     self.decoder_input = TileSeqLast()
        # elif decoder_input=='all':
        #     self.decoder_input = nn.Identity()
        self.decoder_input = Tile()

        self.criterion = nn.L1Loss(reduction="none")
        
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_var = nn.Linear(hidden_dim, latent_dim)


    # def get_encoder(self):
    #     class ModelEncoder(nn.Module):
    #         def __init__(self, encoder, reducer):
    #             super().__init__()
    #             self.encoder = encoder
    #             self.reducer = reducer
            
    #         def forward(self, input_, seq_len):
    #             return self.reducer(self.encoder(input_), seq_len)
    #     return ModelEncoder(self.encoder, self.reduce_encoder_output)

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

        decoder = Decoder(input_dim=self.hparams.hidden_dim, 
                            hidden_dim=self.hparams.hidden_dim, 
                            output_dim=self.hparams.input_dim,
                            bidirectional=self.hparams.bidirectional, 
                            cell=self.hparams.cell, 
                            num_layers=self.hparams.num_layers)
        
        for param in list(encoder.parameters()) + list(decoder.parameters()):
            if param.dim() == 2:
                nn.init.xavier_uniform_(param)

        return encoder, decoder

    def get_encoder(self):
        class ModelEncoder(nn.Module):
            def __init__(self, encoder, reducer):
                super().__init__()
                self.encoder = encoder
                self.reducer = reducer
            
            def forward(self, input_, seq_len):
                return self.reducer(self.encoder(input_), seq_len)
        return ModelEncoder(self.encoder, self.reduce_encoder_output)
        # return nn.Sequential(self.encoder, self.reduce_encoder_output)
        
    def get_representations(self, input_, seq_len):
        
        return self.reduce_encoder_output(self.encoder(input_), seq_len=seq_len)

    def sample(self, mu, log_var):
        std = torch.exp(log_var / 2)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()
        return p, q, z

    def forward(self, input_, seq_len, out_len):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        feat = self.encoder(input_) 
        feats = self.reduce_encoder_output(feat, seq_len=seq_len)
        
        mu = self.fc_mu(feats)
        log_var = self.fc_var(feats)
        p, q, z = self.sample(mu, log_var)
        
        rec = self.decoder(self.decoder_input(z, out_len=out_len))

        return z, rec, p, q
        # return z, rec, mu, log_var

    def shared_step(self, batch, batch_idx):
        (input_, seq_len), _ = batch
        
        z, rec, p, q = self(input_=input_, seq_len=seq_len, out_len=input_.shape[1])

        loss_rec = masked_loss(self.criterion, rec, input_, seq_len)

        log_qz = q.log_prob(z)
        log_pz = p.log_prob(z)

        kl = log_qz - log_pz
        kl = kl.mean()

        # z, rec, mu, log_var = self(input_=input_, seq_len=seq_len, out_len=input_.shape[1])

        # loss_rec = masked_loss(self.criterion, rec, input_, seq_len)
        
        # kl =  - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), -1)
        # kl = kl.mean()

        return loss_rec, kl

    def training_step(self, batch, batch_idx):

        loss_rec, kl = self.shared_step(batch, batch_idx)
        
        loss = loss_rec + self.hparams.beta * kl
        
        self.log('train_rec_loss', loss_rec)
        self.log('train_kl', kl)
        self.log('train_loss', loss)
        
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):

        loss_rec, kl = self.shared_step(batch, batch_idx)
        
        loss = loss_rec + self.hparams.beta * kl
        
        results = {
            'val_rec_loss': loss_rec,
            'val_kl': kl,
            'val_loss': loss,
        }

        return results    

    def validation_epoch_end(self, outputs):
        val_rec_loss = mean(outputs, 'val_rec_loss')
        val_kl = mean(outputs, 'val_kl')
        val_loss = mean(outputs, 'val_loss')

        self.log('val_rec_loss_agg', val_rec_loss)
        self.log('val_kl_agg', val_kl)
        self.log('val_loss_agg', val_loss)

    def test_step(self, batch, batch_idx):

        loss_rec, kl = self.shared_step(batch, batch_idx)
        
        loss = loss_rec + self.hparams.beta * kl
        
        results = {
            'test_rec_loss': loss_rec.cpu(),
            'test_kl': kl.cpu(),
            'test_loss': loss.cpu(),
        }

        return results

    def test_epoch_end(self, outputs):
        rec_loss = mean(outputs, 'test_rec_loss')
        kl = mean(outputs, 'test_kl')
        loss = mean(outputs, 'test_loss')
        
        results = {
            'test_rec_loss': rec_loss.item(),
            'test_kl': kl.item(),
            'test_loss': loss.item(),
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
        parser.add_argument('--learning_rate', type=float, default=0.03)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--output_feature', type=str, default='last') #tap

        # model parameters
        parser.add_argument('--bidirectional', type=bool, default=False) 
        parser.add_argument('--input_dim', type=int, default=75)
        parser.add_argument('--input_embedding', type=str, default=None)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--cell', type=str, default='LSTM')
        parser.add_argument('--num_layers', type=int, default=1)

        parser.add_argument('--latent_dim', type=int, default=512)
        parser.add_argument('--beta', type=float, default=1.0)

        
        return parser


# class AutoEncTri(AutoEncPC):
      
#     def shared_step(self, batch, batch_idx):

#         ((input_1, seq_len_1) , (input_2, seq_len_2), (ground_truth, seq_len_gt)), _ = batch
        
#         rec, feats = self(input_=torch.cat([input_1,input_2],0) , seq_len=torch.cat([seq_len_1,seq_len_2],0), out_len=ground_truth.shape[1])
#         # rec_2, feats_2 = self(input_=input_2)

#         rec_1 = rec[:rec_1.shape[0]]
#         rec_2 = rec[rec_1.shape[0]:]

#         loss_1gt = masked_loss(self.criterion, rec_1, ground_truth, seq_len_gt)
#         loss_2gt = masked_loss(self.criterion, rec_2, ground_truth, seq_len_gt)
#         loss_12 = masked_loss(self.criterion, rec_1, rec_2, seq_len_gt)
        
#         # loss_1gt = ((rec_1 - ground_truth)**2).mean()
#         # loss_2gt = ((rec_2 - ground_truth)**2).mean()
#         # loss_12 = ((rec_1 - rec_2)**2).mean()
        
#         loss = loss_1gt + loss_2gt + loss_12


#         return loss


def masked_loss(criterion, pred, target, seq_len):
    
    target = target.reshape(pred.shape)
    mask = torch.arange(pred.shape[1]).long().expand(pred.shape[:2]) < torch.Tensor(seq_len).long()[:,None]
    mask = mask.to(pred.device)
    
    loss = criterion(pred, target)
    loss = loss.mean(-1)
    loss = (loss*mask).sum(1)/mask.sum(1)
    loss = loss.mean()
    
    return loss
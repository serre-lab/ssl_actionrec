from argparse import ArgumentParser
from typing import Union
from warnings import warn

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from pl_bolts.metrics import precision_at_k, mean

from models import Encoder, Decoder, TemporalAveragePooling, LastHidden, TileLast, LastSeqHidden, TileSeqLast

from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence


class MocoV2TDSlow(pl.LightningModule):

    def __init__(self,
                 emb_dim: int = 128,
                 num_negatives_BU: int = 65536,
                 num_negatives_TD: int = 1024,
                 maximum_timesteps: int = 50,
                 #  num_td_negatives: int = 512,
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
                 weight_bu: float = 1,
                 weight_td: float = 1,
                 weight_bu_sim: float = 1,
                 weight_td_sim: float = 1,
                 *args, **kwargs):

        super().__init__()
        self.save_hyperparameters()

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q, self.encoder_k, self.decoder_q, self.decoder_k = self.init_encoders()

        self.weight_bu = weight_bu
        self.weight_td = weight_td
        self.weight_bu_sim = weight_bu_sim
        self.weight_td_sim = weight_td_sim

        if use_mlp:  # hack: brute-force replacement
            # dim_mlp = self.encoder_q.output_dim
            self.trans_k = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, emb_dim))
            self.trans_q = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, emb_dim))
        else:
            self.trans_k = nn.Linear(hidden_dim, emb_dim)
            self.trans_q = nn.Linear(hidden_dim, emb_dim)

        if output_feature == 'tap':
            self.reduce_encoder_output = TemporalAveragePooling()  # x [B,T,F]
        elif output_feature == 'last':
            self.reduce_encoder_output = LastHidden()  # x [B,T,F]
        elif output_feature == 'seqlast':
            self.reduce_encoder_output = LastSeqHidden()  # x [B,T,F]

        if decoder_input == 'last':
            self.decoder_input = TileLast()
        elif decoder_input == 'seqlast':
            self.decoder_input = TileSeqLast()
        elif decoder_input == 'all':
            self.decoder_input = nn.Identity()

        for param_q_BU, param_k_BU in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k_BU.data.copy_(param_q_BU.data)  # initialize
            param_k_BU.requires_grad = False  # not update by gradient

        for param_q_TD, param_k_TD in zip(self.decoder_q.parameters(), self.decoder_k.parameters()):
            param_k_TD.data.copy_(param_q_TD.data)  # initialize
            param_k_TD.requires_grad = False  # not update by gradient

        # create the BU queue
        self.register_buffer("queue_BU", torch.randn(emb_dim, num_negatives_BU))
        self.queue_BU = nn.functional.normalize(self.queue_BU, dim=0)

        self.register_buffer("queue_ptr_BU", torch.zeros(1, dtype=torch.long))

        self.k_modules_BU = nn.ModuleList([self.encoder_k, self.trans_k])
        self.q_modules_BU = nn.ModuleList([self.encoder_q, self.trans_q])

        # create the TD queue
        self.register_buffer("queue_TD", torch.randn(num_negatives_TD, maximum_timesteps, input_dim))
        self.queue_TD = nn.functional.normalize(self.queue_TD, dim=0)

        self.register_buffer("queue_ptr_TD", torch.zeros(1, dtype=torch.long))

        self.k_modules_TD = nn.ModuleList([self.decoder_k, self.trans_k])
        self.q_modules_TD = nn.ModuleList([self.decoder_q, self.trans_q])

        self.criterion = nn.L1Loss(reduction="none")

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

        for param in list(encoder_q.parameters()) + list(decoder_k.parameters()):
            if param.dim() == 2:
                nn.init.xavier_uniform_(param)

        return encoder_q, encoder_k, decoder_q, decoder_k

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

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """

        for param_q_BU, param_k_BU in zip(self.q_modules_BU.parameters(), self.k_modules_BU.parameters()):
            em = self.hparams.encoder_momentum
            param_k_BU.data = param_k_BU.data * em + param_q_BU.data * (1. - em)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        #if self.use_ddp or self.use_ddp2:
            #keys = concat_all_gather(keys)
            # keys_td = concat_all_gather(keys_td)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_BU)
        assert self.hparams.num_negatives_BU % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + batch_size) > self.hparams.num_negatives_BU:
            self.queue_BU[:, ptr:ptr + batch_size] = keys.T[:self.hparams.num_negatives_BU - ptr]
            self.queue_BU[:, :ptr + batch_size - self.hparams.num_negatives_BU] = keys.T[
                                                                                  self.hparams.num_negatives_BU - ptr:]
        else:
            self.queue_BU[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.hparams.num_negatives_BU  # move pointer

        self.queue_ptr_BU[0] = ptr

        # ptr = int(self.queue_td_ptr)

        # # replace the keys at ptr (dequeue and enqueue)
        # self.queue[:, :, ptr:ptr + batch_size] = keys_td.T
        # ptr = (ptr + batch_size) % self.hparams.num_negatives  # move pointer

        # self.queue_td_ptr[0] = ptr

    @torch.no_grad()
    def _momentum_update_key_encoder_TD(self):
        """
        Momentum update of the key encoder
        """

        for param_q_TD, param_k_TD in zip(self.q_modules_TD.parameters(), self.k_modules_TD.parameters()):
            em = self.hparams.encoder_momentum
            param_k_TD.data = param_k_TD.data * em + param_q_TD.data * (1. - em)

    def get_representations_TD(self, input_, seq_len):

        return self.reduce_encoder_output(self.encoder_q(input_), seq_len=seq_len)

    def get_q_outputs_TD(self, input_q, seq_len, out_len):

        feats_q = self.encoder_q(input_q)  # queries: NxC
        rec_q = self.decoder_q(self.decoder_input(feats_q, seq_len=seq_len, out_len=out_len))

        return rec_q

    def get_k_outputs_TD(self, input_k, seq_len, out_len):

        feats_k = self.encoder_k(input_k)  # queries: NxC
        rec_k = self.decoder_k(self.decoder_input(feats_k, seq_len=seq_len, out_len=out_len))

        return rec_k

    @torch.no_grad()
    def _dequeue_and_enqueue_TD(self, keys):
        # gather keys before updating queue
        # if self.use_ddp or self.use_ddp2:
        #     keys = concat_all_gather(keys)
        #     keys_td = concat_all_gather(keys_td)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr_TD)
        assert self.hparams.num_negatives_TD % batch_size == 0  # for simplicity
        # keys = keys.reshape([batch_size, -1])
        # replace the keys at ptr (dequeue and enqueue)
        if (ptr + batch_size) > self.hparams.num_negatives_TD:
            self.queue_TD[ptr:ptr + batch_size] = keys[:self.hparams.num_negatives_TD - ptr]
            self.queue_TD[:ptr + batch_size - self.hparams.num_negatives_TD] = keys[
                                                                               self.hparams.num_negatives_TD - ptr:]
        else:
            self.queue_TD[ptr:ptr + batch_size] = keys
        ptr = (ptr + batch_size) % self.hparams.num_negatives_TD  # move pointer

        self.queue_ptr_TD[0] = ptr

    def get_representations(self, input_, seq_len):
        return self.reduce_encoder_output(self.encoder_q(input_), seq_len=seq_len)

    def get_q_outputs(self, input_q):

        feats_q = self.encoder_q(input_q[0])  # queries: NxC
        q = self.trans_q(self.reduce_encoder_output(feats_q, input_q[1]))
        q = nn.functional.normalize(q, dim=1)

        return q, feats_q

    def get_k_outputs(self, input_k):

        feats_k = self.encoder_k(input_k[0])  # queries: NxC
        k = self.trans_k(self.reduce_encoder_output(feats_k, input_k[1]))
        k = nn.functional.normalize(k, dim=1)
        # rec_k = self.decoder(self.decoder_input(feats_k))

        return k, feats_k

    def forward(self, input_q, input_k, ground_truth):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q, feats_q = self.get_q_outputs(input_q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k, feats_k = self.get_k_outputs(input_k)

        rec_q = self.decoder(self.decoder_input(feats_q, input_q[1], ground_truth[0].shape[1]))
        rec_k = self.decoder(self.decoder_input(feats_k, input_k[1], ground_truth[0].shape[1]))

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

    def shared_step(self, batch, batch_idx):

        # (input_1, _, input_2, _), _ = batch
        (input_1, input_2, ground_truth), _ = batch

        k, output, target, rec_k, rec_q = self(input_q=input_1, input_k=input_2, ground_truth=ground_truth)
        nll_bu = F.cross_entropy(output.float(), target.long())

        loss_qgt = masked_loss(self.td_criterion, rec_q, ground_truth[0], ground_truth[1])
        loss_kgt = masked_loss(self.td_criterion, rec_k, ground_truth[0], ground_truth[1])
        loss_qk = masked_loss(self.td_criterion, rec_q, rec_k, ground_truth[1])

        loss_td = loss_qgt + loss_kgt + loss_qk

        loss = self.bu_loss_weight * nll_bu + self.td_loss_weight * loss_td

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        return (loss,
                k,
                acc1,
                acc5,
                nll_bu,
                loss_td,
                loss_qgt,
                loss_kgt,
                loss_qk
                )

    def training_step(self, batch, batch_idx):

        (loss,
         k,
         acc1,
         acc5,
         nll_bu,
         loss_td,
         loss_qgt,
         loss_kgt,
         loss_qk
         ) = self.shared_step(batch, batch_idx)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        self.log('train_mse_q_gt', loss_qgt)
        self.log('train_mse_k_gt', loss_kgt)
        self.log('train_mse_q_k', loss_qk)
        self.log('train_bu_loss', nll_bu)
        self.log('train_td_loss', loss_td)
        self.log('train_acc1', acc1)
        self.log('train_acc5', acc5)

        return {'loss': loss}  # , 'log': log, 'progress_bar': log

    def validation_step(self, batch, batch_idx):

        (_,
         _,
         acc1,
         acc5,
         nll_bu,
         loss_td,
         _,
         _,
         _
         ) = self.shared_step(batch, batch_idx)

        results = {
            'val_bu_loss': nll_bu,
            'val_td_loss': loss_td,
            'val_acc1': acc1,
            'val_acc5': acc5
        }

        return results

    def validation_epoch_end(self, outputs):
        val_bu_loss = mean(outputs, 'val_bu_loss')
        val_td_loss = mean(outputs, 'val_td_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')

        self.log('val_bu_loss_agg', val_bu_loss)
        self.log('val_td_loss_agg', val_td_loss)
        self.log('val_acc1_agg', val_acc1)
        self.log('val_acc5_agg', val_acc5)

    def test_step(self, batch, batch_idx):

        (_,
         _,
         acc1,
         acc5,
         nll_bu,
         loss_td,
         _,
         _,
         _
         ) = self.shared_step(batch, batch_idx)

        results = {
            'test_bu_loss': nll_bu.cpu(),
            'test_td_loss': loss_td.cpu(),
            'test_acc1': acc1.cpu(),
            'test_acc5': acc5.cpu(),
        }

        return results

    def test_epoch_end(self, outputs):
        test_bu_loss = mean(outputs, 'test_bu_loss')
        test_td_loss = mean(outputs, 'test_td_loss')
        test_acc1 = mean(outputs, 'test_acc1')
        test_acc5 = mean(outputs, 'test_acc5')

        results = {
            'test_bu_loss': test_loss.item(),
            'test_td_loss': test_loss.item(),
            'test_con_acc1': test_acc1.item(),
            'test_con_acc5': test_acc5.item(),
        }

        return results

    def configure_optimizers(self):
        if self.hparams.optimizer == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), self.hparams.learning_rate,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.weight_decay)

            eta_min = self.hparams.learning_rate * (self.hparams.lr_decay_rate ** 3)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.epochs, eta_min, -1)

            # lamda_fun = lambda epoch: self.hparams.lr_decay_rate ** np.sum(epoch > np.asarray([30,70]))
            # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lamda_fun)
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
        parser.add_argument('--num_negatives_BU', type=int, default=16000)
        parser.add_argument('--num_negatives_TD', type=int, default=1024)
        # parser.add_argument('--num_td_negatives', type=int, default=65536)
        parser.add_argument('--encoder_momentum', type=float, default=0.999)
        parser.add_argument('--softmax_temperature', type=float, default=0.07)
        parser.add_argument('--learning_rate', type=float, default=0.03)
        parser.add_argument('--momentum', type=float, default=0.9)
        parser.add_argument('--weight_decay', type=float, default=1e-4)
        parser.add_argument('--use_mlp', action='store_true')
        parser.add_argument('--output_feature', type=str, default='seqlast')  # tap
        parser.add_argument('--decoder_input', type=str, default='seqlast')  # all

        parser.add_argument('--epochs', type=int, default=60)
        parser.add_argument('--lr_decay_rate', type=float, default=0.1)

        # model parameters
        parser.add_argument('--bidirectional', type=bool, default=False)
        parser.add_argument('--input_dim', type=int, default=75)
        parser.add_argument('--input_embedding', type=str, default=None)
        parser.add_argument('--hidden_dim', type=int, default=128)
        parser.add_argument('--cell', type=str, default='LSTM')
        parser.add_argument('--num_layers', type=int, default=1)

        parser.add_argument('--weight_bu', type=float, default=1)
        parser.add_argument('--weight_td', type=float, default=1)
        parser.add_argument('--weight_bu_sim', type=float, default=1)
        parser.add_argument('--weight_td_sim', type=float, default=1)

        return parser


##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################

class MocoPlusAE(MocoV2TDSlow):

    def forward(self, input_q, input_k, ground_truth):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        q, feats_q = self.get_q_outputs(input_q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k, _ = self.get_k_outputs(input_k)

        gt, feats_gt = self.get_q_outputs(ground_truth)
        # print(ground_truth[0].shape[1])
        rec_gt = self.decoder(self.decoder_input(feats_gt, seq_len=ground_truth[1], out_len=ground_truth[0].shape[1]))

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue_BU.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        cos_gt = 2 - 2 * (q * gt).sum(-1)

        return k, logits, labels, rec_gt, cos_gt

    def shared_step(self, batch, batch_idx):
        # (input_1, _, input_2, _), _ = batch
        (input_1, input_2, ground_truth), _ = batch

        k, output, target, rec_gt, cos_gt = self(input_q=input_1, input_k=input_2, ground_truth=ground_truth)
        nll_bu = F.cross_entropy(output.float(), target.long())
        loss_td = masked_loss(self.td_criterion, rec_gt, ground_truth[0], ground_truth[1])
        cos_gt = cos_gt.mean()

        loss = self.bu_loss_weight * nll_bu + self.td_loss_weight * loss_td + self.sim_loss_weight * cos_gt

        acc1, acc5 = precision_at_k(output, target, top_k=(1, 5))

        return (loss,
                k,
                acc1,
                acc5,
                nll_bu,
                loss_td,
                cos_gt,
                )

    def training_step(self, batch, batch_idx):
        (loss,
         k,
         acc1,
         acc5,
         loss_bu,
         loss_td_con,
         sim_bu,
         loss_td_sim,
         ) = self.shared_step(batch, batch_idx)

        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        self.log('val_loss', loss)
        self.log('val_bu_att_loss', sim_bu)
        self.log('val_bu_con_loss', loss_bu)
        self.log('val_td_con_loss', loss_td_con)
        self.log('val_td_att_loss', loss_td_sim)
        self.log('val_acc1', acc1)
        self.log('val_acc5', acc5)

        return {'loss': loss}  # , 'log': log, 'progress_bar': log

    def validation_step(self, batch, batch_idx):
        (loss,
         k,
         acc1,
         acc5,
         loss_bu,
         loss_td_con,
         sim_bu,
         loss_td_sim,
         ) = self.shared_step(batch, batch_idx)

        results = {
            'val_loss': loss,
            'val_bu_att_loss': sim_bu,
            'val_bu_con_loss': loss_bu,
            'val_td_con_loss': loss_td_con,
            'val_td_att_loss': loss_td_sim,
            'val_acc1': acc1,
            'val_acc5': acc5
        }

        return results

    def validation_epoch_end(self, outputs):
        val_bu_con_loss = mean(outputs, 'val_bu_con_loss')
        val_bu_att_loss = mean(outputs, 'val_bu_att_loss')
        val_td_con_loss = mean(outputs, 'val_td_con_loss')
        val_td_att_loss = mean(outputs, 'val_td_att_loss')
        val_acc1 = mean(outputs, 'val_acc1')
        val_acc5 = mean(outputs, 'val_acc5')

        self.log('val_bu_att_loss_agg', val_bu_att_loss)
        self.log('val_bu_con_loss_agg', val_bu_con_loss)
        self.log('val_td_con_loss_agg', val_td_con_loss)
        self.log('val_td_att_loss_agg', val_td_att_loss)
        self.log('val_acc1_agg', val_acc1)
        self.log('val_acc5_agg', val_acc5)

    def test_step(self, batch, batch_idx):
        (loss,
         k,
         acc1,
         acc5,
         loss_bu,
         loss_td_con,
         sim_bu,
         loss_td_sim,
         ) = self.shared_step(batch, batch_idx)

        results = {
            'test_loss': loss,
            'test_bu_att_loss': sim_bu,
            'test_bu_con_loss': loss_bu,
            'test_td_con_loss': loss_td_con,
            'test_td_att_loss': loss_td_sim,
            'test_acc1': acc1,
            'test_acc5': acc5
        }

        return results

    def test_epoch_end(self, outputs):
        test_bu_con_loss = mean(outputs, 'test_bu_con_loss')
        test_bu_att_loss = mean(outputs, 'test_bu_att_loss')
        test_td_con_loss = mean(outputs, 'test_td_con_loss')
        test_td_att_loss = mean(outputs, 'test_td_att_loss')
        test_acc1 = mean(outputs, 'test_acc1')
        test_acc5 = mean(outputs, 'test_acc5')


        results = {
            'test_bu_con_loss': test_bu_con_loss.item(),
            'test_bu_att_loss': test_bu_att_loss.item(),
            'test_td_con_loss': test_td_con_loss.item(),
            'test_td_att_loss': test_td_att_loss.item(),
            'test_acc1': test_acc1.item(),
            'test_acc5': test_acc5.item(),
        }

        return results



##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################
##############################################################################################################################################################################################################################################

class MocoBUTDcon(MocoPlusAE):

    def forward(self, input_q, input_k, ground_truth):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        ########### BU contrastive
        # compute query features
        q, feats_q = self.get_q_outputs(input_q)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k, feats_k = self.get_k_outputs(input_k)

        # compute gt features
        gt, feats_gt = self.get_q_outputs(ground_truth)
        # print(ground_truth[0].shape[1])

        # compute logits
        # Einstein sum is more intuitive
        # positive logits: Nx1
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        # negative logits: NxK
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue_BU.clone().detach()])

        # logits: Nx(1+K)
        logits = torch.cat([l_pos, l_neg], dim=1)

        # apply temperature
        logits /= self.hparams.softmax_temperature

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long)
        labels = labels.type_as(logits)

        sim_bu = 2 - 2 * (q * gt).sum(-1)

        ########### TD contrastive
        out_len = ground_truth[0].shape[1]
        # compute query features
        rec_q = self.decoder_q(self.decoder_input(feats_q, seq_len=input_q[1], out_len=out_len))

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder_TD()  # update the key encoder

            rec_k = self.decoder_k(self.decoder_input(feats_k, seq_len=input_k[1], out_len=out_len))

        err_pos = masked_loss_TD(self.criterion, rec_q, rec_k, ground_truth[1], reduction='none')
        err_neg = masked_loss_TD(self.criterion, rec_q[:, None], self.queue_TD[None, :], ground_truth[1],
                                 reduction='none')

        logits_pos = - err_pos / self.hparams.softmax_temperature
        logits_neg = - err_neg / self.hparams.softmax_temperature

        logits_td = torch.cat([logits_pos[:, None], logits_neg], dim=1)

        labels_td = torch.zeros(logits_td.shape[0], dtype=torch.long)
        labels_td = labels_td.type_as(logits_td)

        rec_gt = self.decoder_q(self.decoder_input(feats_gt, seq_len=ground_truth[1], out_len=ground_truth[0].shape[1]))

        return k, logits, labels, rec_gt, rec_q, rec_k, logits_td, labels_td, sim_bu

    def shared_step(self, batch, batch_idx):
        # (input_1, _, input_2, _), _ = batch
        (input_1, input_2, ground_truth), _ = batch

        k, output_BU, target_BU, rec_gt, rec_q, rec_k, output_TD, target_TD, sim_bu = self(input_q=input_1,
                                                                                           input_k=input_2,
                                                                                           ground_truth=ground_truth)

        loss_bu = F.cross_entropy(output_BU.float(), target_BU.long())

        gt = ground_truth[0].view(list(ground_truth[0].shape[:2]) + [-1])

        loss_td_con = F.cross_entropy(output_TD.float(), target_TD.long())
        # loss_td_sim = masked_loss(self.criterion, rec_q, gt, ground_truth[1])
        loss_td_sim = masked_loss(self.criterion, rec_gt, gt, ground_truth[1])

        loss = self.weight_bu * loss_bu + self.weight_td * loss_td_con + self.weight_bu_sim * sim_bu.mean() + self.weight_td_sim * loss_td_sim

        #print(loss_bu)
        #print(loss_td_con)
        #print(sim_bu.mean())
        #print(loss_td_sim)
        #print(loss)

        acc1, acc5 = precision_at_k(output_BU, target_BU, top_k=(1, 5))

        return (loss,
                k,
                acc1,
                acc5,
                loss_bu,
                loss_td_con,
                sim_bu.mean(),
                loss_td_sim,
                )

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


def masked_loss(criterion, pred, target, seq_len):
    target = target.reshape(pred.shape)
    mask = torch.arange(pred.shape[1]).long().expand(pred.shape[:2]) < torch.Tensor(seq_len).long()[:, None]
    mask = mask.to(pred.device)

    loss = criterion(pred, target)
    loss = loss.mean(-1)
    loss = (loss * mask).sum(1) / mask.sum(1)
    loss = loss.mean()

    return loss


def pad_to_size(t, size, dim):
    pad_list = [0] * 2 * len(t.shape)
    dim = (len(t.shape) - dim - 1) % len(t.shape)
    pad_list[dim * 2 + 1] = size
    return F.pad(t, pad_list, 'constant')


def masked_loss_TD(criterion, pred, target, seq_len, reduction='mean', pad=True):
    if pad:
        pred = pad_to_size(pred, target.shape[-2] - pred.shape[-2], dim=-2)

    target = target.reshape(list(target.shape[:-2]) + list(pred.shape[-2:]))
    mask = torch.arange(pred.shape[-2]).long().expand(pred.shape[:-1])

    mask = mask < torch.Tensor(seq_len).long().view([-1] + [1] * (len(mask.shape) - 1))

    mask = mask.to(pred.device)

    pred = pred - target
    loss = criterion(pred, torch.zeros_like(pred))
    loss = loss.mean(-1)
    loss = (loss * mask).sum(-1) / mask.sum(-1)

    if reduction == 'mean':
        loss = loss.mean()

    return loss
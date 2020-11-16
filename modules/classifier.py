from argparse import ArgumentParser
from typing import Union
from warnings import warn

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

from pl_bolts.metrics import precision_at_k, mean

from models import Encoder, TemporalAveragePooling, LastHidden, TileLast

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
        
class LinearClassifierMod(pl.LightningModule):

    def __init__(self,
                 input_dim: int = 128,
                 n_label: int = 60,
                 learning_rate: float = 1,
                 momentum: float = 0.9,
                 nesterov: bool = True,
                 weight_decay: float = 0,
                 epochs: int = 90,
                 lr_decay_rate: float = 0.5,
                 *args, **kwargs):

        super().__init__()
        self.save_hyperparameters()

        self.classifier = nn.Linear(input_dim, n_label)
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)
        
    def forward(self, input_):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """

        # compute query features
        logits = self.classifier(input_)

        return logits

    def training_step(self, batch, batch_idx):

        input_, label = batch
        logits = self(input_=input_)
        loss = F.cross_entropy(logits.float(), label.long())
        
        acc1, acc5 = precision_at_k(logits, label, top_k=(1, 5))

        self.log('train_loss', loss)
        self.log('train_acc1', acc1)
        self.log('train_acc5', acc5)
        
        return {'loss': loss} 

    def validation_step(self, batch, batch_idx):

        input_, label = batch
        logits = self(input_=input_)
        loss = F.cross_entropy(logits.float(), label.long())
        
        acc1, acc5 = precision_at_k(logits, label, top_k=(1, 5))
        
        results = {
            'val_loss': loss,
            'val_acc1': acc1,
            'val_acc5': acc5,
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

        input_, label = batch
        logits = self(input_=input_)
        loss = F.cross_entropy(logits.float(), label.long())
        
        acc1, acc5 = precision_at_k(logits, label, top_k=(1, 5))
        
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

        return {
            'test_lincls_loss': test_loss.item(),
            'test_lincls_acc1': test_acc1.item(),
            'test_lincls_acc5': test_acc5.item(),
            }

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), self.hparams.learning_rate,
                                    momentum=self.hparams.momentum,
                                    # nesterov=self.hparams.nesterov,
                                    weight_decay=self.hparams.weight_decay)
        # eta_min = self.hparams.learning_rate * (self.hparams.lr_decay_rate ** 3) * 0.1
        eta_min = self.hparams.learning_rate * (self.hparams.lr_decay_rate ** 1) # * 0.1
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.hparams.epochs, eta_min, -1)
        
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [35,45], gamma=self.hparams.lr_decay_rate, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [15,35,60,75], gamma=0.5, last_epoch=-1)

        
        return [optimizer], [scheduler]
    

def knn(data_train, data_test, label_train, label_test, nn=1):
    label_train = np.asarray(label_train)
    label_test = np.asarray(label_test)

    Xtr_Norm = preprocessing.normalize(data_train)
    Xte_Norm = preprocessing.normalize(data_test)

    knn = KNeighborsClassifier(
        n_neighbors=nn, metric="cosine"
    )  # , metric='cosine'#'mahalanobis', metric_params={'V': np.cov(data_train)})
    knn.fit(Xtr_Norm, label_train)
    
    pred = knn.predict(Xte_Norm)
    acc = accuracy_score(pred, label_test)
    return acc

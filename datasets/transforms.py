

from datasets.tools import *
import numpy as np

class NTUMocoTDTrainTransformsTS:
    def __init__(self):
        # super().__init__()

        self.transform = transform_lib.Compose([
            TemporalFlip(),
            Shear(),
            Center(),
            ToTensor(),
        ])
        self.gt_transform = transform_lib.Compose([
            Center(),
            ToTensor(),
        ])
    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        gt = self.gt_transform(inp)
        return q, k, gt

NTUMocoTDEvalTransformsTS = NTUMocoTDTrainTransformsTS


class NTUMocoTDTrainTransformsTSMany:
    def __init__(self, n_trans=16):
        # super().__init__()
        self.n_trans=n_trans
        self.transform = transform_lib.Compose([
            TemporalFlip(),
            Shear(),
            Center(),
            ToTensor(),
        ])
        self.gt_transform = transform_lib.Compose([
            Center(),
            ToTensor(),
        ])
    def __call__(self, inp):
        ret = tuple([self.transform(inp) for i in range(self.n_trans-1)])
        gt = self.gt_transform(inp)
        return ret + (gt,)

NTUMocoTDEvalTransformsTSMany = NTUMocoTDTrainTransformsTSMany

##############################################################################################################
##############################################################################################################
##############################################################################################################

class UCLATDConTransformsTS:
    """
    Moco 2 augmentation:
    https://arxiv.org/pdf/2003.04297.pdf
    """

    def __init__(self):

        # image augmentation functions
        self.transform = transform_lib.Compose([
            TemporalFlip(),
            Shear(),
            ToTensor(),
        ])
        self.gt_transform = transform_lib.Compose([
            ToTensor(),
        ])


    def __call__(self, inp):
        q = self.transform(inp)
        k = self.transform(inp)
        gt = self.gt_transform(inp)
        # print(q.shape)
        return q, k, gt

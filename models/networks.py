from __future__ import print_function

import torch
import torch.nn as nn
from torch.nn import LSTM, RNN, GRU
from torchvision import transforms

# from feeders.tools import aug_look, ToTensor


class TemporalAveragePooling(nn.Module):
    
    def __init__(self):
        super(TemporalAveragePooling, self).__init__()
    
    def forward(self, x, *args, **kwargs):
        return x.mean(1)

class LastHidden(nn.Module):
    
    def __init__(self):
        super(LastHidden, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x[:,-1]

class LastSeqHidden(nn.Module):
    
    def __init__(self):
        super(LastSeqHidden, self).__init__()

    def forward(self, x, seq_len, *args, **kwargs):
        return x[torch.arange(x.shape[0]),torch.Tensor(seq_len).long()-1]

class TileSeqLast(nn.Module):
    
    def __init__(self):
        super(TileSeqLast, self).__init__()

    def forward(self, x, seq_len, out_len, *args, **kwargs):
        return torch.stack([x[torch.arange(x.shape[0]),torch.Tensor(seq_len).long()-1]]*out_len, 1) 

class TileLast(nn.Module):
    
    def __init__(self):
        super(TileLast, self).__init__()

    def forward(self, x, out_len, *args, **kwargs):
        return torch.stack([x[:,-1]]*out_len, 1) 

class Tile(nn.Module):
    
    def __init__(self):
        super(Tile, self).__init__()

    def forward(self, x, out_len, *args, **kwargs):
        return torch.stack([x]*out_len, 1) 

class LinearClassifier(nn.Module):
    def __init__(self,  last_layer_dim=None, n_label=None, ):
        super(LinearClassifier, self).__init__()

        self.classifier = nn.Linear(last_layer_dim, n_label)
        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.classifier(x)


# def reshap_input_for_lstm(x1):
#     n, c, t, v, m = x1.size()
#     x1 = x1.permute(2, 0, 1, 3, 4).contiguous()  # t, n, c, v, m
#     x1 = x1.view(t, n, -1)
#     return x1

def reshap_input_for_lstm(x1):
    n, t, m, v, c = x1.size()
    x1 = x1.view(n, t, -1)
    return x1

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, bidirectional=False, embedding=None, cell='LSTM', num_layers=1):
        super(Encoder, self).__init__()
        if embedding =='mlp':
            self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.Tanh())
            rnn_input = hidden_dim
        else:
            self.embedding = nn.Identity()
            rnn_input = input_dim
        
        if bidirectional:
            hidden_dim = hidden_dim//2

        if cell == 'LSTM':
            self.rnn = LSTM(input_size=rnn_input, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        elif cell == 'GRU':
            self.rnn = GRU(input_size=rnn_input, hidden_size=hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

        # self.rnn.flatten_parameters()

    def forward(self, seq):
        if len(seq.shape)>3:
            seq = reshap_input_for_lstm(seq)
        seq = seq.float()
        # t, b, f = seq.shape
        b, t, f = seq.shape
        
        seq = self.embedding(seq.reshape([-1, f])).reshape([b, t, -1])

        seq, _ = self.rnn(seq)
        # seq = seq.permute(1, 0, 2).contiguous()  # n,b,h
        
        return seq


class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, bidirectional=False, cell='LSTM', num_layers=1):
        super(Decoder, self).__init__()
        
        rnn_hidden_dim = hidden_dim//2 if bidirectional else hidden_dim
        
        if cell == 'LSTM':
            self.rnn = LSTM(input_size=input_dim, hidden_size=rnn_hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        elif cell == 'GRU':
            self.rnn = GRU(input_size=input_dim, hidden_size=rnn_hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)

        self.rnn.flatten_parameters()

        rnn_hidden_output = rnn_hidden_dim * 2 if bidirectional else hidden_dim
        
        self.output_lin = nn.Linear(rnn_hidden_output, output_dim)
        # self.output_lin = nn.Sequential(nn.Linear(rnn_hidden_output, output_dim), nn.Tanh())

    def forward(self, seq):
        # seq = seq.permute(1, 0, 2).contiguous()  # t,b,h
        seq, _ = self.rnn(seq)
        
        # seq = seq.permute(1, 0, 2).contiguous()  # b,t,h
        
        # t, b, f = seq.shape
        b, t, f = seq.shape
        output = self.output_lin(seq.reshape([-1, f])).reshape([b, t, -1])
        return output


class Head(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, flag):
        super(Head, self).__init__()
        self.flag = flag
        if flag == 'linear':
            self.model = nn.Sequential(nn.Linear(input_dim, output_dim, ))
        elif flag == 'nonlinear':
            self.model = nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Linear(hidden_dim, output_dim, )
                                       )
        else:
            raise NotImplementedError("not option")

        self.initilize()

    def initilize(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                # if self.flag == 'nonlinear':
                #     m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.model(x)


# def aug_transfrom(aug_name, args_list, norm, norm_aug, args):
#     aug_name_list = aug_name.split("_")
#     transform_aug = [aug_look('selectFrames', args.selected_frames)]

#     if aug_name_list[0] != 'None':
#         for i, aug in enumerate(aug_name_list):
#                 transform_aug.append(aug_look(aug, args_list[i * 2], args_list[i * 2 + 1]))

#     if norm == 'normalizeC':
#         #TODO remove normalizeC and normalizeCV from everywhere
#         print("hello")
#         #transform_aug.extend([Skeleton2Image(), ToTensor(), norm_aug, Image2skeleton()])
#     elif norm == 'normalizeCV':
#         transform_aug.extend([ToTensor(), norm_aug])
#     else:
#         transform_aug.extend([ToTensor(), ])
#     transform_aug = transforms.Compose(transform_aug)

#     return transform_aug
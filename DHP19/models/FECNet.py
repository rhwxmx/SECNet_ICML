import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import LocalGrouper_P as LocalGrouper
import math
from typing import Any, Callable, List, Optional, Union
from numbers import Number
import numpy as np
import argparse
class Linear1Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True):
        super(Linear1Layer, self).__init__()
        # self.act = nn.ReLU(inplace=True)
        self.act = nn.GELU()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)

class Linear2Layer(nn.Module):
    def __init__(self, in_channels, kernel_size=1, groups=1, bias=True):
        super(Linear2Layer, self).__init__()

        # self.act = nn.ReLU(inplace=True)
        self.act = nn.GELU()
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=int(in_channels/2),
                    kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(in_channels/2)),
            self.act
        )

        self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(in_channels/2), out_channels=in_channels,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(in_channels)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        attn_weights = self.linear(lstm_output).squeeze(-1)
        attn_probs = torch.softmax(attn_weights, dim=1)
        return attn_probs

class fftlayer(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.complex_weight = weight
    def forward(self, x):
        x = torch.fft.rfft(x, dim=(1), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x = x * weight
        x = torch.fft.irfft(x, dim=(1), norm='ortho')
        return x

class FECNet(nn.Module):
    def __init__(self,args):
        super().__init__()
        self.args = args
        self.sizeH = args.sensor_sizeH
        self.sizeW = args.sensor_sizeW
        self.num_joints = args.num_joints
        self.feature_list = [64,132,268,540]
        self.group_number = [512,256,128,64]         
        self.neighbors = [24,24,24,24]
        self.stages = 3
        self.local_grouper_list = nn.ModuleList()
        self.fft_temporal_weight_list = nn.ParameterList()
        self.fft_sptaial_weight_list = nn.ParameterList() 
        self.fft_temporal_list = nn.ModuleList()
        self.fft_sptaial_list = nn.ModuleList()   
        self.embed_dim = Linear1Layer(4,self.feature_list[0],1)
        self.group_conv_list = nn.ModuleList()
        self.aggregation_list = nn.ModuleList()
        self.conv_list = nn.ModuleList()
        self.attention = Attention(self.feature_list[-1])
        for i in range(self.stages):
            local_grouper = LocalGrouper(self.feature_list[i], self.group_number[i], 24, True, "anchor")
            self.local_grouper_list.append(local_grouper)
            aggregation = Attention(self.feature_list[i+1])
            self.aggregation_list.append(aggregation)
            fft_sptaial_weight = nn.Parameter(torch.cat((torch.ones(self.feature_list[i+1]//2+1, self.neighbors[i],1, dtype=torch.float32),torch.zeros(self.feature_list[i+1]//2+1, self.neighbors[i],1, dtype=torch.float32)),dim=-1))
            self.fft_sptaial_list.append(fftlayer(fft_sptaial_weight))
            fft_temporal_weight = nn.Parameter(torch.cat((torch.ones(self.group_number[i]//2+1, self.feature_list[i+1],1, dtype=torch.float32),torch.zeros(self.group_number[i]//2+1, self.feature_list[i+1],1, dtype=torch.float32)),dim=-1))
            self.fft_temporal_list.append(fftlayer(fft_temporal_weight))
            conv = Linear2Layer(self.feature_list[i+1],1,1)
            self.conv_list.append(conv)

        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(self.num_joints * 128)
        self.dp1 = nn.Dropout(p=0.1)
        self.dp2 = nn.Dropout(p=0.1)
        self.out_conv1 = nn.Sequential(nn.Linear(self.feature_list[self.stages], 512),
                                       self.bn6,
                                       nn.ReLU(),
                                       self.dp1)
        self.out_conv2 = nn.Sequential(nn.Linear(512, self.num_joints * 128),
                                       self.bn7,
                                       nn.ReLU(),
                                       self.dp2)
        
        self.mlp_head_x = nn.Linear(128, self.sizeW)
        self.mlp_head_y = nn.Linear(128, self.sizeH)

    def forward(self, x: torch.Tensor):
        xyz = x.permute(0,2,1)
        batch_size, _, _ = x.size()
        x = self.embed_dim(x)
        x = x.permute(0,2,1)
        for i in range(self.stages):
            xyz, x = self.local_grouper_list[i](xyz, x)
            x= x.permute(0, 1, 3, 2)
            b, n, d, s = x.size()
            x = x.reshape(-1,d,s)
            x = self.fft_sptaial_list[i](x)
            x = F.gelu(x) 
            x = x.permute(0,2,1)
            att = self.aggregation_list[i](x)
            x = torch.bmm(att.unsqueeze(1), x).squeeze(1)
            x = x.reshape(b, n, -1)
            x = self.fft_temporal_list[i](x)
            x = x.permute(0,2,1)
            x = F.gelu(x)  
            x = self.conv_list[i](x)
            x = x.permute(0,2,1)

        attn = self.attention(x)
        x = torch.bmm(attn.unsqueeze(1), x).squeeze(1)
        
        x = self.out_conv1(x)
        x = self.out_conv2(x)
        x = x.view(batch_size, self.num_joints, -1)
        pred_x = self.mlp_head_x(x)
        pred_y = self.mlp_head_y(x)
        return pred_x, pred_y

def rfft_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the rfft/rfftn operator.
    """
    input_shape = inputs[0].type().sizes()
    B, N, C = input_shape
    flops = N * C * np.ceil(np.log2(N))+N*C*2
    flops = flops * B
    return flops

if __name__ == '__main__':

    input = torch.randn(1, 4, 4096)
    from thop import profile
    from thop import clever_format
    from torchinfo import summary
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import torchprofile
    import time
    argsparser = argparse.ArgumentParser()
    args = argsparser.parse_args()
    args.sensor_sizeH = 260
    args.sensor_sizeW = 346
    args.num_joints = 13
    model =  FECNet(args)
    model.eval()
    fca1 = FlopCountAnalysis(model, input)
    handlers = {
        'aten::fft_rfft': rfft_flop_jit,
        'aten::fft_irfft': rfft_flop_jit,
    }
    fca1.set_op_handle(**handlers)
    flops1 = fca1.total()
    print(flop_count_table(fca1, max_depth=1))
    print("#### GMACs: {}".format(flops1 / 1e9/2))

        
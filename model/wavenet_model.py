import os
import os.path
import time
from data.dataset import *
import torch.nn as nn
import torch.nn.functional as F

class WaveNetModel(nn.Module):

    def __init__(self, hparams, device):

        super(WaveNetModel, self).__init__()

        self.layers = hparams.layers
        self.blocks = hparams.blocks
        self.dilation_channels = hparams.dilation_channels
        self.residual_channels = hparams.residual_channels
        self.skip_channels = hparams.skip_channels
        self.input_channel = hparams.input_channel
        self.initial_kernel = hparams.initial_kernel
        self.kernel_size = hparams.kernel_size
        self.output_channel = hparams.output_channel
        self.bias = hparams.bias

        self.device = device
        # build model
        receptive_field = 1
        init_dilation = 1

        self.dilated_convs = nn.ModuleList()

        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()

        # 1x1 convolution to create channels
        self.start_conv = nn.Conv1d(in_channels=self.input_channel,
                                    out_channels=self.residual_channels,
                                    kernel_size=self.initial_kernel,
                                    bias=self.bias)



        for b in range(self.blocks):
            additional_scope = self.kernel_size - 1
            new_dilation = 1
            actual_layer = self.layers
            if b == self.blocks-1:
                actual_layer = self.layers - 1
            for i in range(actual_layer):

                # dilated convolutions
                self.dilated_convs.append(nn.Conv1d(in_channels=self.residual_channels,
                                                    out_channels=self.dilation_channels*2,
                                                    kernel_size=self.kernel_size,
                                                    bias=self.bias,
                                                    dilation=new_dilation))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                     out_channels=self.residual_channels,
                                                     kernel_size=1,
                                                     bias=self.bias))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=self.dilation_channels,
                                                 out_channels=self.skip_channels,
                                                 kernel_size=1,
                                                 bias=self.bias))

                receptive_field += additional_scope
                additional_scope *= 2
                init_dilation = new_dilation
                new_dilation *= 2

        self.end_conv = nn.Conv1d(in_channels=self.skip_channels,
                                  out_channels=self.output_channel,
                                  kernel_size=1,
                                  bias=self.bias)



        self.receptive_field = receptive_field + self.initial_kernel - 1

    def forward(self, input):

        x = self.start_conv(input)
        skip = 0

        # WaveNet layers
        for i in range(self.blocks * self.layers - 1):

            # |----------------------------------------------------|     *residual*
            # |                                                    |
            # |                |---- tanh --------|                |
            # -> dilate_conv ->|                  * ----|-- 1x1 -- + -->	*input*
            #                  |---- sigm --------|     |
            #                                          1x1
            #                                           |
            # ----------------------------------------> + ------------->	*skip*

            residual = x

            dilated = self.dilated_convs[i](x)

            filter, gate = torch.chunk(dilated, 2, dim=1)

            # dilated convolution
            filter = torch.tanh(filter)
            gate = torch.sigmoid(gate)
            x = filter * gate

            # parametrized skip connection
            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, -s.size(2):]
            except:
                skip = 0
            skip = s + skip

            x = self.residual_convs[i](x)
            x = x + residual[:, :, -x.size(2):]

        x = torch.relu_(skip)
        x = self.end_conv(x)

        return x

    def parameter_count(self):
        par = list(self.parameters())
        s = sum([np.prod(list(d.size())) for d in par])
        return s

    def get_phonetic(self, input):

        real_length = input.shape[1]
        to_pad = int(self.receptive_field / 2)
        data = np.pad(input, ((0, 0), (to_pad, to_pad)), 'constant', constant_values=0)

        preds = []
        raw_out = []
        for i in range(real_length):
            model_input = data[:, i:i+self.receptive_field]
            model_input = torch.FloatTensor(model_input).unsqueeze(0).to(self.device)
            out = self.forward(model_input)
            raw_out.append(out.detach().squeeze())
            pred = out.max(1)[1]
            preds.append(pred.squeeze())

        return preds, raw_out
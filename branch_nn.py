import torch
from torch import nn

from resnet import ResBlock1d
from resnet import _downsample
from resnet import _padding
import torch.nn.functional as F


class DNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DNN, self).__init__()
        self.linears = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, 512))
        self.linears.append(nn.Linear(512, 512))
        self.linears.append(nn.Linear(512, output_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.linears[0](h))
        h = F.relu(self.linears[1](h))
        return self.linears[2](h)


class ResNet1d(nn.Module):
    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8):
        super(ResNet1d, self).__init__()
        # First layers
        n_filters_in, n_filters_out = input_dim[0], blocks_dim[0][0]
        n_samples_in, n_samples_out = input_dim[1], blocks_dim[0][1]
        downsample = _downsample(n_samples_in, n_samples_out)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, bias=False,
            stride=downsample, padding=padding)
        self.bn1 = nn.BatchNorm1d(n_filters_out)

        # Residual block layers
        self.res_blocks = []
        for i, (n_filters, n_samples) in enumerate(blocks_dim):
            n_filters_in, n_filters_out = n_filters_out, n_filters
            n_samples_in, n_samples_out = n_samples_out, n_samples
            downsample = _downsample(n_samples_in, n_samples_out)
            resblk1d = ResBlock1d(n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate)
            self.add_module('resblock1d_{0}'.format(i), resblk1d)
            self.res_blocks += [resblk1d]

    def forward(self, x):
        """Implement ResNet1d forward propagation"""
        # First layers
        x = self.conv1(x)
        x = self.bn1(x)

        # Residual blocks
        y = x
        for blk in self.res_blocks:
            x, y = blk(x, y)

        # Flatten array
        x = x.view(x.size(0), -1)
        return x


class Branch_NN(nn.Module):
    def __init__(self, dnn_input_dim, dnn_output_dim, cnn_input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8):
        super(Branch_NN, self).__init__()
        self.cnn = ResNet1d(
            input_dim=cnn_input_dim,
            blocks_dim=blocks_dim,
            n_classes=n_classes,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate
        )
        self.dnn = DNN(dnn_input_dim, dnn_output_dim)
        n_filters_last, n_samples_last = blocks_dim[-1]
        cnn_last_layer_dim = n_filters_last * n_samples_last
        last_layer_dim = dnn_output_dim + cnn_last_layer_dim
        self.dense_layer = nn.Linear(last_layer_dim, n_classes)

    def forward(self, cnn_x, dnn_x):
        cnn_out = self.cnn(cnn_x)
        dnn_out = self.dnn(dnn_x)
        out = torch.cat([cnn_out, dnn_out], dim=1)
        return self.dense_layer(out)

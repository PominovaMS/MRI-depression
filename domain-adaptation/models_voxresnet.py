import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Identity(nn.Module):
    def __init__(self,):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x


class Reshape(nn.Module):
    def __init__(self, shape):
        nn.Module.__init__(self)
        self.shape = shape
    def forward(self, input):
        return input.view((-1,) + self.shape)


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class VoxResNet(nn.Module):
    def __init__(self, input_shape=(128, 128, 128), num_classes=2, n_filters=32, stride=2, n_blocks=3, n_flatten_units=None, dropout=0, n_fc_units=128):
        super(self.__class__, self).__init__()
        self.model = nn.Sequential()

        self.model.add_module("conv3d_1", nn.Conv3d(1, n_filters, kernel_size=3, padding=1, stride=stride)) # n * (x/s) * (y/s) * (z/s)
        self.model.add_module("batch_norm_1", nn.BatchNorm3d(n_filters))
        self.model.add_module("activation_1", nn.ReLU(inplace=True))
        self.model.add_module("conv3d_2", nn.Conv3d(n_filters, n_filters, kernel_size=3, padding=1)) # n * (x/s) * (y/s) * (z/s)
        self.model.add_module("batch_norm_2", nn.BatchNorm3d(n_filters))
        self.model.add_module("activation_2", nn.ReLU(inplace=True))

#         1
        self.model.add_module("conv3d_3", nn.Conv3d(n_filters, 2 * n_filters, kernel_size=3, padding=1, stride=2)) # 2n * (x/2s) * (y/2s) * (z/2s)
        self.model.add_module("block_1", BasicBlock(2 * n_filters, 2 * n_filters))
        self.model.add_module("block_2", BasicBlock(2 * n_filters, 2 * n_filters))
        self.model.add_module("batch_norm_3", nn.BatchNorm3d(2 * n_filters))
        self.model.add_module("activation_3", nn.ReLU(inplace=True))

#         2
        if n_blocks >= 2:
            self.model.add_module("conv3d_4", nn.Conv3d(2 * n_filters, 2 * n_filters, kernel_size=3, padding=1, stride=2)) # 2n * (x/4s) * (y/4s) * (z/4s)
            self.model.add_module("block_3", BasicBlock(2 * n_filters, 2 * n_filters))
            self.model.add_module("block_4", BasicBlock(2 * n_filters, 2 * n_filters))
            self.model.add_module("batch_norm_4", nn.BatchNorm3d(2 * n_filters))
            self.model.add_module("activation_4", nn.ReLU(inplace=True))

#         3
        if n_blocks >= 3:
            self.model.add_module("conv3d_5", nn.Conv3d(2 * n_filters, 4 * n_filters, kernel_size=3, padding=1, stride=2)) # 4n * (x/8s) * (y/8s) * (z/8s)
            self.model.add_module("block_5", BasicBlock(4 * n_filters, 4 * n_filters))
            self.model.add_module("block_6", BasicBlock(4 * n_filters, 4 * n_filters))
            self.model.add_module("batch_norm_5", nn.BatchNorm3d(4 * n_filters))
            self.model.add_module("activation_5", nn.ReLU(inplace=True))

#         4
        if n_blocks >= 4:
            self.model.add_module("conv3d_6", nn.Conv3d(4 * n_filters, 4 * n_filters, kernel_size=3, padding=1, stride=2)) # 4n * (x/16s) * (y/16s) * (z/16s)
            self.model.add_module("block_7", BasicBlock(4 * n_filters, 4 * n_filters))
            self.model.add_module("block_8", BasicBlock(4 * n_filters, 4 * n_filters))
            self.model.add_module("batch_norm_6", nn.BatchNorm3d(4 * n_filters))
            self.model.add_module("activation_6", nn.ReLU(inplace=True))

#         self.model.add_module("max_pool3d_1", nn.MaxPool3d(kernel_size=3)) # (b/2)n * (x/(2^b)sk) * (y/(2^b)sk) * (z/(2^b)sk) ?
        
        if n_flatten_units is None:
            n_flatten_units = 4 * n_filters * np.prod(np.array(input_shape) // (2 ** n_blocks * stride))
        print(n_flatten_units)
        
        self.model.add_module("flatten_1", Flatten())
        self.model.add_module("fully_conn_1", nn.Linear(n_flatten_units, n_fc_units))
        self.model.add_module("activation_6", nn.ReLU(inplace=True))
        self.model.add_module("dropout_1", nn.Dropout(dropout))

        self.model.add_module("fully_conn_2", nn.Linear(n_fc_units, num_classes))

            
    def forward(self, x):
        return self.model(x)
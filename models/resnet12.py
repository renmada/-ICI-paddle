import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import nn as nn
import numpy as np


class DropBlock(nn.Layer):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()

        self.block_size = block_size
        offsets = paddle.stack(
            [
                paddle.arange(self.block_size).reshape([-1, 1]).expand([self.block_size, self.block_size]).reshape(
                    [-1]),
                paddle.arange(self.block_size).tile([self.block_size]),
            ]
        ).t()
        self.offsets = paddle.concat((paddle.zeros([self.block_size ** 2, 2]).astype('int64'), offsets.astype('int64')), 1)

    def forward(self, x, gamma):

        if self.training:
            batch_size, channels, height, width = x.shape
            bernoulli = lambda shape: paddle.to_tensor(np.random.random(shape) < gamma).astype('float32')
            mask = bernoulli((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1)))
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.shape[0] * block_mask.shape[1] * block_mask.shape[2] * block_mask.shape[3]
            count_ones = block_mask.sum()
            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):

        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)

        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        # offsets = paddle.stack(
        #     [
        #         paddle.arange(self.block_size).reshape([-1, 1]).expand([self.block_size, self.block_size]).reshape([-1]),
        #         paddle.arange(self.block_size).tile([self.block_size]),
        #     ]
        # ).t()
        # offsets = paddle.concat((paddle.zeros([self.block_size ** 2, 2]).astype('int64'), offsets.astype('int64')), 1)
        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.tile([self.block_size ** 2, 1])
            offsets = self.offsets.tile([nr_blocks, 1]).reshape([-1, 4])
            offsets = offsets.astype('int64')
            block_idxs = non_zero_idxs + offsets
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask = padded_mask.cpu().numpy()
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
            padded_mask = paddle.to_tensor(padded_mask)
        else:
            padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))

        block_mask = 1 - padded_mask
        return block_mask


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias_attr=False)


class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1,
                 pool=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2D(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2D(planes)
        self.maxpool = nn.MaxPool2D(stride, ceil_mode=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)
        self.pool = pool

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if self.pool:
            out = self.maxpool(out)

        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.shape[2]
                keep_rate = max(1.0 - self.drop_rate / (20 * 2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size ** 2 * feat_size ** 2 / (feat_size - self.block_size + 1) ** 2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training)

        return out


class ResNet(nn.Layer):

    def __init__(self, block, drop_block=False, drop_rate=0.1, dropblock_size=5):
        self.inplanes = 3
        self.nFeat = 512
        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, stride=2, drop_rate=drop_rate, drop_block=drop_block,
                                       block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 512, stride=2, drop_rate=drop_rate, drop_block=drop_block,
                                       block_size=dropblock_size, pool=True)


    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1, pool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2D(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias_attr=False),
                nn.BatchNorm2D(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size, pool))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        return x4


def resnet12(drop_block=True, **kwargs):
    """Constructs a ResNet-12 model.
    """
    model = ResNet(BasicBlock, drop_block=drop_block, **kwargs)
    return model

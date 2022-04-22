from __future__ import absolute_import
from __future__ import division

import paddle
import paddle.nn as nn


class CrossEntropyLoss(nn.Layer):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.logsoftmax = nn.LogSoftmax(1)

    def forward(self, inputs, targets):
        inputs = inputs.reshape([inputs.shape[0], inputs.shape[1], -1])
        log_probs = self.logsoftmax(inputs)
        zeros = paddle.zeros([inputs.shape[0], inputs.shape[1]])
        targets = targets.unsqueeze(1)
        targets = paddle.put_along_axis(zeros, targets, paddle.ones_like(targets).astype('float32'), 1)
        # targets = nn.functional.one_hot(targets, inputs.shape[1])
        targets = targets.unsqueeze(-1)
        loss = (- targets * log_probs).mean(0).sum()
        return loss / inputs.shape[2]

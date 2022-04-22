from __future__ import absolute_import

import paddle


def init_optimizer(optim, params, lr, weight_decay):
    if optim == 'adam':
        return paddle.optimizer.Adam(parameters=params, learning_rate=lr, weight_decay=weight_decay)
    # elif optim == 'amsgrad':
    #     return paddle.optimizer.Adam(parameters=params, learning_rate=lr, weight_decay=weight_decay, amsgrad=True)
    elif optim == 'sgd':
        return paddle.optimizer.Momentum(parameters=params, learning_rate=lr, momentum=0.9, weight_decay=weight_decay)
    elif optim == 'rmsprop':
        return paddle.optimizer.RMSProp(parameters=params, learning_rate=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise KeyError("Unsupported optimizer: {}".format(optim))

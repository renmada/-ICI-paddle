import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from models.resnet12 import resnet12
import time


class Model(nn.Layer):
    def __init__(self, scale_cls, iter_num_prob=35.0 / 75, num_classes=64):
        super(Model, self).__init__()
        self.scale_cls = scale_cls
        self.iter_num_prob = iter_num_prob
        self.base = resnet12()
        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2D(self.nFeat, num_classes, kernel_size=1)

    def get_embeddings(self, x):
        f = self.base(x)
        f = f.mean(2).mean(2)
        f = F.normalize(f, p=2, axis=f.dim() - 1, epsilon=1e-12)
        return f

    def test(self, ftrain, ftest):
        ftrain = ftrain.mean(3).mean(3)
        ftest = ftest.mean(3).mean(3)
        ftest = F.normalize(ftest, p=2, axis=ftest.dim() - 1, epsilon=1e-12)
        ftrain = F.normalize(ftrain, p=2, axis=ftrain.dim() - 1, epsilon=1e-12)
        scores = self.scale_cls * paddle.matmul(ftest, ftrain, transpose_y=True)
        return scores

    def forward(self, xtrain, xtest, ytrain, ytest):
        batch_size, num_train = xtrain.shape[0], xtrain.shape[1]
        num_test = xtest.shape[1]
        K = ytrain.shape[2]
        ytrain = ytrain.transpose([0, 2, 1])

        xtrain = xtrain.reshape([-1, xtrain.shape[2], xtrain.shape[3], xtrain.shape[4]])
        xtest = xtest.reshape([-1, xtest.shape[2], xtest.shape[3], xtest.shape[4]])
        x = paddle.concat((xtrain, xtest), 0)
        f = self.base(x)
        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.reshape([batch_size, num_train, -1])
        ftrain = paddle.matmul(ytrain, ftrain)
        ftrain = ftrain / (ytrain.sum(2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.reshape([batch_size, -1, *f.shape[1:]])
        ftest = f[batch_size * num_train:]
        ftest = ftest.reshape([batch_size, num_test, *f.shape[1:]])
        if not self.training:
            return self.test(ftrain, ftest)

        b, n1, c, h, w = ftrain.shape
        n2 = ftest.shape[1]
        ftrain = ftrain.reshape([b, n1, c, -1])
        ftest = ftest.reshape([b, n2, c, -1])
        ftrain = ftrain.unsqueeze(2).tile([1, 1, n2, 1, 1])
        ftrain = ftrain.reshape([b, n1, n2, c, h, w]).transpose([0, 2, 1, 3, 4, 5])
        ftest = ftest.unsqueeze(1).tile([1, 1, n1, 1, 1])
        ftest = ftest.reshape([b, n1, n2, c, h, w]).transpose([0, 2, 1, 3, 4, 5])

        ftrain = ftrain.mean(4).mean(4)

        ftest_norm = F.normalize(ftest, p=2, axis=3, epsilon=1e-12)
        ftrain_norm = F.normalize(ftrain, p=2, axis=3, epsilon=1e-12)
        ftrain_norm = ftrain_norm.unsqueeze(4)
        ftrain_norm = ftrain_norm.unsqueeze(5)
        cls_scores = self.scale_cls * paddle.sum(ftest_norm * ftrain_norm, axis=3)
        cls_scores = cls_scores.reshape([batch_size * num_test, *cls_scores.shape[2:]])

        ftest = ftest.reshape([batch_size, num_test, K, -1])
        ftest = ftest.transpose([0, 1, 3, 2])
        ytest = ytest.unsqueeze(3)
        ftest = paddle.matmul(ftest, ytest)
        ftest = ftest.reshape([batch_size * num_test, *f.shape[1:]])
        ytest = self.clasifier(ftest)
        return ytest, cls_scores

    def predict(self, x):
        f = self.base(x)
        f = self.clasifier(f)
        return f.mean([2, 3])


class InferModel(nn.Layer):
    def __init__(self, scale_cls, iter_num_prob=35.0 / 75, num_classes=64):
        super(InferModel, self).__init__()
        self.scale_cls = scale_cls
        self.iter_num_prob = iter_num_prob
        self.base = resnet12()
        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2D(self.nFeat, num_classes, kernel_size=1)

    def forward(self, x):
        f = self.base(x)
        f = self.clasifier(f)
        f = f.mean([2, 3])
        return F.softmax(f, axis=1)

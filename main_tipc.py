import datetime
import math
import os
import os.path as osp
import sys
import time
import paddle.nn.functional as F

import numpy as np
import paddle
from tqdm import tqdm

from config import config
from data.data_manager import DataManager
from models.ici import ICI
from models.net import Model
from utils.avgmeter import AverageMeter
from utils.ci import mean_confidence_interval
from utils.iotools import save_checkpoint
from utils.logger import Logger
from utils.losses import CrossEntropyLoss
from utils.optimizers import init_optimizer
from utils.tools import one_hot
from paddle.optimizer.lr import PiecewiseDecay


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices

    sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    print('Initializing image data manager')
    dm = DataManager(args)
    trainloader, testloader = dm.return_dataloaders()

    model = Model(scale_cls=args.scale_cls, num_classes=args.num_classes)
    criterion = CrossEntropyLoss()
    lr_schedule = PiecewiseDecay([60, 70, 80, 90], [0.1, 0.006, 0.0012, 0.00024])
    optimizer = init_optimizer(args.optim, model.parameters(), lr_schedule, args.weight_decay)

    if args.resume is not None:
        state_dict = paddle.load(args.resume)['state_dict']
        model.set_state_dict(state_dict)
        print('Load model from {}'.format(args.resume))

    if args.mode == 'test':
        test(model, testloader, args)
        return

    best_acc = -np.inf
    best_epoch = 0
    print("==> Start training")

    global_step = 0
    train_reader_cost = 0.0
    train_run_cost = 0.0
    total_samples = 0
    reader_start = time.time()
    last_step = args.max_epoch * len(trainloader)
    for epoch in range(args.start_epoch, args.max_epoch):
        for images_train, labels_train, images_test, labels_test, pids in trainloader:
            labels_train_1hot = one_hot(labels_train)
            labels_test_1hot = one_hot(labels_test)
            train_reader_cost += time.time() - reader_start
            global_step += 1
            train_start = time.time()
            ytest, cls_scores = model(images_train, images_test, labels_train_1hot, labels_test_1hot)
            loss1 = criterion(ytest, pids.reshape([-1]))
            loss2 = criterion(cls_scores, labels_test.reshape([-1]))
            loss = loss1 + 0.5 * loss2
            train_run_cost += time.time() - train_start
            total_samples += len(images_train)
            if global_step % 10 == 0:
                print(
                    "global step %d / %d, loss: %f, avg_reader_cost: %.5f sec, avg_batch_cost: %.5f sec, avg_samples: %.5f, ips: %.5f img/sec"
                    % (global_step, last_step, loss.item(), train_reader_cost /
                       10, (train_reader_cost + train_run_cost)
                       / 10, total_samples / 10,
                       total_samples / (train_reader_cost + train_run_cost)))
                train_reader_cost = 0.0
                train_run_cost = 0.0
                total_samples = 0
                reader_start = time.time()
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()

        acc = val(model, testloader)
        is_best = acc > best_acc

        if is_best:
            best_acc = acc
            best_epoch = epoch + 1

        state_dict = model.state_dict()
        print(state_dict.keys())
        state_dict.pop('clasifier.weight')
        state_dict.pop('clasifier.bias')
        save_checkpoint(state_dict, is_best, osp.join(args.save_dir, 'checkpoint_ep' + str(epoch + 1) + '.tar'))

        print("==> Test 5-way Best accuracy {:.2%}, achieved at epoch {}".format(best_acc, best_epoch))

    print("==========\nArgs:{}\n==========".format(args))


def val(model, testloader):
    accs = AverageMeter()
    test_accuracies = []
    model.eval()

    with paddle.no_grad():
        for images_train, labels_train, images_test, labels_test, _ in testloader:
            batch_size = images_train.shape[0]
            num_test_examples = images_test.shape[1]

            labels_train_1hot = one_hot(labels_train)
            labels_test_1hot = one_hot(labels_test)

            cls_scores = model(images_train, images_test, labels_train_1hot, labels_test_1hot)
            cls_scores = cls_scores.reshape([batch_size * num_test_examples, -1])
            labels_test = labels_test.reshape([batch_size * num_test_examples])

            preds = cls_scores.argmax(1)
            # acc = (paddle.sum(preds == labels_test).float()) / labels_test.shape[0]
            acc = (preds == labels_test).astype('float32').mean()
            accs.update(acc.item(), labels_test.shape[0])

            gt = (preds == labels_test).astype('float32')
            gt = gt.reshape([batch_size, num_test_examples]).cpu().numpy()
            acc = np.sum(gt, 1) / num_test_examples
            acc = np.reshape(acc, (batch_size))
            test_accuracies.append(acc)

    accuracy = accs.avg
    test_accuracies = np.array(test_accuracies)
    test_accuracies = np.reshape(test_accuracies, -1)
    mean_acc, ci = mean_confidence_interval(test_accuracies)
    print('Accuracy: {:.2%}, std: :{:.2%}'.format(mean_acc, ci))

    return accuracy


def test(model, testloader, args):
    import warnings
    warnings.filterwarnings('ignore')
    ici = ICI(classifier=args.classifier, num_class=args.nKnovel,
              step=args.step, strategy=args.strategy, reduce=args.embed,
              d=args.dim, logit_penalty=args.logit_penalty)
    if args.unlabel != 0:
        iterations = math.ceil(args.unlabel / args.step) + 2
    else:
        iterations = math.ceil(15 / args.step) + 2
    acc_list = [[] for _ in range(iterations)]

    model.eval()
    with paddle.no_grad():
        for images_train, labels_train, images_test, labels_test, images_unlabel in tqdm(testloader, ncols=0):

            assert images_train.shape[0] == 1

            num_train = images_train.shape[1]
            num_test = images_test.shape[1]
            if args.unlabel != 0:
                images = paddle.concat([images_train, images_test, images_unlabel], 1).squeeze(0)
            else:
                images = paddle.concat([images_train, images_test], 1).squeeze(0)
            embeddings = model.get_embeddings(images).detach().cpu().numpy()
            train_embeddings = embeddings[:num_train]
            labels_train = labels_train.squeeze(0).numpy().reshape(-1)
            test_embeddings = embeddings[num_train:num_train + num_test]
            labels_test = labels_test.squeeze(0).numpy().reshape(-1)
            if args.unlabel != 0:
                unlabel_embeddings = embeddings[num_train + num_test:]
            else:
                unlabel_embeddings = None

            ici.fit(train_embeddings, labels_train)
            acc = ici.predict(test_embeddings, unlabel_embeddings, True, labels_test)
            for i in range(min(iterations - 1, len(acc))):
                acc_list[i].append(acc[i])
            acc_list[-1].append(acc[-1])

    mean_acc_list = []
    ci_list = []
    for i, item in enumerate(acc_list):
        mean_acc, ci = mean_confidence_interval(item)
        mean_acc_list.append(mean_acc)
        ci_list.append(ci)
    print("{}".format(
        ' '.join([str(i * 100)[:5] for i in mean_acc_list])))
    print("{}".format(' '.join([str(i * 100)[:5] for i in ci_list])))

    return


if __name__ == '__main__':
    args = config()
    main(args)

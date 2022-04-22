# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
from paddle import inference
import numpy as np
import sys
import os
import paddle
import paddle.nn.functional as F
import numpy as np
import argparse
from paddle.io import DataLoader
from sklearn.metrics import f1_score, accuracy_score
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

from utils.ci import mean_confidence_interval



class InferenceEngine(object):
    """InferenceEngine
    Inference engina class which contains preprocess, run, postprocess
    """

    def __init__(self, args):
        """
        Args:
            args: Parameters generated using argparser.
        Returns: None
        """
        super().__init__()
        self.args = args

        # init inference engine
        self.predictor, self.config, self.input_tensors, self.output_tensors = self.load_predictor(
            os.path.join(args.model_dir, "inference.pdmodel"),
            os.path.join(args.model_dir, "inference.pdiparams"))

    def load_predictor(self, model_file_path, params_file_path):
        """load_predictor
        initialize the inference engine
        Args:
            model_file_path: inference model path (*.pdmodel)
            model_file_path: inference parmaeter path (*.pdiparams)
        Return:
            predictor: Predictor created using Paddle Inference.
            config: Configuration of the predictor.
            input_tensor: Input tensor of the predictor.
            output_tensor: Output tensor of the predictor.
        """
        args = self.args
        config = inference.Config(model_file_path, params_file_path)
        if args.device == "gpu":
            config.enable_use_gpu(1000, 0)
        else:
            config.disable_gpu()

        # enable memory optim
        config.enable_memory_optim()
        config.disable_glog_info()

        config.switch_use_feed_fetch_ops(False)
        config.switch_ir_optim(True)

        # create predictor
        predictor = inference.create_predictor(config)

        # get input and output tensor property
        input_tensors = [
            predictor.get_input_handle(name)
            for name in predictor.get_input_names()
        ]
        output_tensors = [
            predictor.get_output_handle(name)
            for name in predictor.get_output_names()
        ]

        return predictor, config, input_tensors, output_tensors

    def preprocess(self, args):
        """preprocess
        Preprocess to the input.
        Args:
            data: data.
        Returns: Input data after preprocess.
        """
        dm = DataManager(args)
        trainloader, testloader = dm.return_dataloaders()
        return testloader

    def postprocess(self, output):
        """postprocess
        Postprocess to the inference engine output.
        Args:
            output: Inference engine output.
        Returns: Output data after argmax.
        """
        return output

    def run(self, data):
        """run
        Inference process using inference engine.
        Args:
            x: Input data after preprocess.
        Returns: Inference engine output
        """
        self.input_tensors[0].copy_from_cpu(data)
        self.predictor.run()
        output = self.output_tensors[0].copy_to_cpu()
        return output


def infer_main(args):
    """infer_main
    Main inference function.
    Args:
        args: Parameters generated using argparser.
    Returns:
        label_id: Class index of the input.
        prob: : Probability of the input.
    """
    inference_engine = InferenceEngine(args)
    ici = ICI(classifier=args.classifier, num_class=args.nKnovel,
              step=args.step, strategy=args.strategy, reduce=args.embed,
              d=args.dim, logit_penalty=args.logit_penalty)
    if args.unlabel != 0:
        iterations = math.ceil(args.unlabel / args.step) + 2
    else:
        iterations = math.ceil(15 / args.step) + 2
    acc_list = [[] for _ in range(iterations)]

    # init benchmark
    if args.benchmark:
        import auto_log
        autolog = auto_log.AutoLogger(
            model_name="icir",
            batch_size=1,
            inference_config=inference_engine.config,
            gpu_ids="auto" if args.device=='gpu' else None)

    # enable benchmark
    if args.benchmark:
        autolog.times.start()

    # dataset preprocess
    test_loader = inference_engine.preprocess(args)
    if args.benchmark:
        autolog.times.stamp()

    for images_train, labels_train, images_test, labels_test, images_unlabel in tqdm(test_loader, ncols=0):
        assert images_train.shape[0] == 1

        num_train = images_train.shape[1]
        num_test = images_test.shape[1]
        if args.unlabel != 0:
            images = paddle.concat([images_train, images_test, images_unlabel], 1).squeeze(0)
        else:
            images = paddle.concat([images_train, images_test], 1).squeeze(0)
        embeddings = inference_engine.run(images.cpu().numpy())
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

    if args.benchmark:
        autolog.times.stamp()

    # postprocess
    mean_acc_list = []
    ci_list = []
    for i, item in enumerate(acc_list):
        mean_acc, ci = mean_confidence_interval(item)
        mean_acc_list.append(mean_acc)
        ci_list.append(ci)
    print("{}".format(
        ' '.join([str(i * 100)[:5] for i in mean_acc_list])))
    print("{}".format(' '.join([str(i * 100)[:5] for i in ci_list])))

    if args.benchmark:
        autolog.times.stamp()
        autolog.times.end(stamp=True)
        autolog.report()


if __name__ == "__main__":
    args = config()
    infer_main(args)

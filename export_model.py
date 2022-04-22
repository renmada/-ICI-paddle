# -*- coding: UTF-8 -*-
#   Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import argparse
import os

import paddle
from paddle.static import InputSpec
from config import config
from models.net import InferModel

# yapf: disable
args = config()


# yapf: enable


def main():
    model = InferModel(scale_cls=args.scale_cls, num_classes=args.num_classes)

    state_dict = paddle.load(args.resume)
    model.set_dict(state_dict)
    model.eval()

    model = paddle.jit.to_static(
        model,
        input_spec=[InputSpec(shape=[None, 3, 84, 84], dtype="float32", name='x')]
    )
    # Save in static graph model.
    paddle.jit.save(model, os.path.join(args.output_path, "inference"))


if __name__ == "__main__":
    main()

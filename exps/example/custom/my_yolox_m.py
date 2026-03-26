#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # Define yourself dataset path
        self.data_dir = "C:/Users/gyeongmin/Desktop/YOLO/data/package"
        self.train_ann = "C:/Users/gyeongmin/Desktop/YOLO/260325/train.json"
        self.val_ann = "C:/Users/gyeongmin/Desktop/YOLO/260325/val.json"

        self.num_classes = 1

        self.max_epoch = 500
        self.data_num_workers = 8
        self.eval_interval = 1

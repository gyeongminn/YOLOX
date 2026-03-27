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
        self.data_num_workers = 4
        self.eval_interval = 1

        # FP 억제: 마지막 no_aug 구간을 늘려 정밀 수렴 시간 확보
        self.no_aug_epochs = 30  # 기본 15 → 30

        # ── 처음 보는 자재 일반화 ──────────────────────────────────────
        # 기하학: 더 넓은 회전/원근 변형 → 다양한 각도/거리의 자재 학습
        self.degrees   = 30.0          # 기본 10° → 30° (다양한 각도)
        self.shear     = 8.0           # 기본 2°  → 8°  (원근 왜곡)
        self.translate = 0.2           # 기본 0.1 → 0.2 (위치 이동)

        # 외관: 색상·질감 의존도 낮추기 → 형태/엣지 기반 학습 유도
        self.grayscale_prob = 0.15     # 15% 확률로 그레이스케일
        self.blur_prob      = 0.15     # 15% 확률로 가우시안 블러

    def get_model(self):
        model = super().get_model()
        # obj_loss 가중치 상향 → 배경에서 높은 confidence 출력 시 더 강한 페널티
        model.head.obj_weight = 2.0
        return model

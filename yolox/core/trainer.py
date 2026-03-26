#!/usr/bin/env python3
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.exp import Exp
from yolox.utils import (
    MeterBuffer,
    MlflowLogger,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    mem_usage,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)


class Trainer:
    def __init__(self, exp: Exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.amp.GradScaler("cuda", enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)

        # figure 생성용 히스토리
        self._loss_history = {
            "total_loss": [], "iou_loss": [], "conf_loss": [], "cls_loss": [], "l1_loss": []
        }
        self._ap_history = {"ap50_95": [], "ap50": []}
        self._lr_history = []
        self._epoch_loss_accum = {k: [] for k in self._loss_history}
        self._epoch_loss_accum["lr"] = []
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception as e:
            logger.error("Exception in training: ", e)
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        data_end_time = time.time()

        with torch.amp.autocast("cuda", enabled=self.amp_training):
            outputs = self.model(inps, targets)

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

        # 에폭 평균 계산용 누적
        for k in ["total_loss", "iou_loss", "conf_loss", "cls_loss", "l1_loss"]:
            if k in outputs:
                self._epoch_loss_accum[k].append(float(outputs[k]))
        self._epoch_loss_accum["lr"].append(float(lr))

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                self.wandb_logger = WandbLogger.initialize_wandb_logger(
                    self.args,
                    self.exp,
                    self.evaluator.dataloader.dataset
                )
            elif self.args.logger == "mlflow":
                self.mlflow_logger = MlflowLogger()
                self.mlflow_logger.setup(args=self.args, exp=self.exp)
            else:
                raise ValueError("logger must be either 'tensorboard', 'mlflow' or 'wandb'")

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()
            elif self.args.logger == "mlflow":
                metadata = {
                    "epoch": self.epoch + 1,
                    "input_size": self.input_size,
                    'start_ckpt': self.args.ckpt,
                    'exp_file': self.args.exp_file,
                    "best_ap": float(self.best_ap)
                }
                self.mlflow_logger.on_train_end(self.args, file_name=self.file_name,
                                                metadata=metadata)

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))

        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        self.save_ckpt(ckpt_name="latest")

        # 에폭 평균 손실 기록
        for k in self._loss_history:
            vals = self._epoch_loss_accum[k]
            self._loss_history[k].append(sum(vals) / len(vals) if vals else 0.0)
        lr_vals = self._epoch_loss_accum["lr"]
        self._lr_history.append(lr_vals[-1] if lr_vals else 0.0)
        for k in self._epoch_loss_accum:
            self._epoch_loss_accum[k].clear()

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

        # 10 에폭마다 figure 저장
        if self.rank == 0 and (self.epoch + 1) % 10 == 0:
            self._save_training_figures()

    def before_iter(self):
        pass

    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            mem_str = "gpu mem: {:.0f}Mb, mem: {:.1f}Gb".format(gpu_mem_usage(), mem_usage())

            logger.info(
                "{}, {}, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    mem_str,
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )

            if self.rank == 0:
                if self.args.logger == "tensorboard":
                    self.tblogger.add_scalar(
                        "train/lr", self.meter["lr"].latest, self.progress_in_iter)
                    for k, v in loss_meter.items():
                        self.tblogger.add_scalar(
                            f"train/{k}", v.latest, self.progress_in_iter)
                if self.args.logger == "wandb":
                    metrics = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    metrics.update({
                        "train/lr": self.meter["lr"].latest
                    })
                    self.wandb_logger.log_metrics(metrics, step=self.progress_in_iter)
                if self.args.logger == 'mlflow':
                    logs = {"train/" + k: v.latest for k, v in loss_meter.items()}
                    logs.update({"train/lr": self.meter["lr"].latest})
                    self.mlflow_logger.on_log(self.args, self.exp, self.epoch+1, logs)

            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device, weights_only=False)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        with adjust_status(evalmodel, training=False):
            (ap50_95, ap50, summary), predictions = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed, return_outputs=True
            )

        update_best_ckpt = ap50_95 > self.best_ap
        self.best_ap = max(self.best_ap, ap50_95)

        # AP 히스토리 기록
        self._ap_history["ap50_95"].append(float(ap50_95))
        self._ap_history["ap50"].append(float(ap50))

        # figure 생성용 평가 데이터 캡처 (rank 0에서만 의미 있음)
        if self.rank == 0:
            coco_eval = getattr(self.evaluator, "_last_coco_eval", None)
            coco_gt   = getattr(self.evaluator, "_last_coco_gt",   None)
            prec   = coco_eval.eval.get("precision") if coco_eval else None
            scores = coco_eval.eval.get("scores")    if coco_eval else None
            self._fig_eval_data = {
                "precision":   prec,
                "scores":      scores,
                "output_data": predictions,   # image_id → {bboxes, scores, categories}
                "coco_gt":     coco_gt,
            }

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "train/epoch": self.epoch + 1,
                })
                self.wandb_logger.log_images(predictions)
            if self.args.logger == "mlflow":
                logs = {
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "val/best_ap": round(self.best_ap, 3),
                    "train/epoch": self.epoch + 1,
                }
                self.mlflow_logger.on_log(self.args, self.exp, self.epoch+1, logs)
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            cur = self.epoch + 1
            if cur == 1 or cur % 10 == 0:
                self.save_ckpt(f"epoch_{cur}", ap=ap50_95)

        if self.args.logger == "mlflow":
            metadata = {
                    "epoch": self.epoch + 1,
                    "input_size": self.input_size,
                    'start_ckpt': self.args.ckpt,
                    'exp_file': self.args.exp_file,
                    "best_ap": float(self.best_ap)
                }
            self.mlflow_logger.save_checkpoints(self.args, self.exp, self.file_name, self.epoch,
                                                metadata, update_best_ckpt)

    # ──────────────────────────────────────────────────────────────────────
    # Figure 생성 헬퍼
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _box_iou_xyxy(a, b):
        """두 bbox (xyxy) 간 IoU 계산"""
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        return inter / (area_a + area_b - inter + 1e-6)

    def _load_img(self, img_id, coco_gt, dataset):
        """이미지 numpy array 로드 (실패 시 None 반환)"""
        try:
            from PIL import Image as PILImage
            img_info = coco_gt.loadImgs(img_id)[0]
            fn = img_info.get("file_name", "")
            data_dir = getattr(dataset, "data_dir", "")
            name     = getattr(dataset, "name", "")
            for candidate in [
                os.path.join(data_dir, name, fn),
                os.path.join(data_dir, fn),
                fn,
            ]:
                if os.path.exists(candidate):
                    return img_info, __import__("numpy").array(PILImage.open(candidate).convert("RGB"))
        except Exception:
            pass
        return None, None

    # ── 메인 figure 저장 ────────────────────────────────────────────────
    def _save_training_figures(self):
        import numpy as np

        epoch_num  = self.epoch + 1
        fig_dir    = os.path.join(self.file_name, "figures", f"epoch_{epoch_num:04d}")
        os.makedirs(fig_dir, exist_ok=True)

        epochs     = list(range(1, epoch_num + 1))
        ap_epochs  = list(range(1, len(self._ap_history["ap50_95"]) + 1))
        eval_data  = getattr(self, "_fig_eval_data", {}) or {}
        prec_arr   = eval_data.get("precision")   # (T,R,K,A,M)
        scores_arr = eval_data.get("scores")       # (T,R,K,A,M)
        output_data = eval_data.get("output_data") # img_id→{bboxes,scores,categories}
        coco_gt    = eval_data.get("coco_gt")
        dataset    = self.evaluator.dataloader.dataset
        r_thresh   = np.linspace(0, 1, 101)

        def _savefig(fig, name):
            fig.savefig(os.path.join(fig_dir, name), dpi=120, bbox_inches="tight")
            plt.close(fig)

        # ── 01 손실 곡선 ─────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        fig.suptitle(f"Loss Curves — {self.exp.exp_name}  Epoch {epoch_num}", fontsize=12, fontweight="bold")
        for (key, title, color), ax in zip(
            [("total_loss", "Total Loss", "steelblue"),
             ("iou_loss",   "Box (IoU) Loss", "tomato"),
             ("conf_loss",  "Obj (Conf) Loss", "darkorange"),
             ("cls_loss",   "Class Loss", "mediumseagreen")],
            axes.flat
        ):
            ax.plot(epochs, self._loss_history[key], color=color, linewidth=1.5)
            if self._loss_history["l1_loss"] and key == "total_loss":
                ax.plot(epochs, self._loss_history["l1_loss"], color="mediumpurple",
                        linewidth=1, linestyle="--", label="l1_loss")
                ax.legend(fontsize=8)
            ax.axvline(epoch_num, color="gray", linestyle="--", alpha=0.35, linewidth=1)
            ax.set_title(title, fontsize=10); ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
            ax.grid(True, alpha=0.3)
        plt.tight_layout(); _savefig(fig, "01_loss_curves.png")

        # ── 02 Learning Rate ─────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, self._lr_history, color="slateblue", linewidth=1.5)
        ax.set_title(f"Learning Rate Schedule — Epoch {epoch_num}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch"); ax.set_ylabel("LR")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(True, alpha=0.3)
        plt.tight_layout(); _savefig(fig, "02_lr_curve.png")

        # ── 03 AP 곡선 ───────────────────────────────────────────────────
        if ap_epochs:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Validation mAP — Epoch {epoch_num}", fontsize=12, fontweight="bold")
            for ax, key, color, label in [
                (axes[0], "ap50_95", "royalblue",      "mAP@0.5:0.95"),
                (axes[1], "ap50",    "mediumseagreen", "mAP@0.50"),
            ]:
                vals = [v * 100 for v in self._ap_history[key]]
                ax.plot(ap_epochs, vals, color=color, linewidth=1.5, marker=".", markersize=4)
                bv = max(vals); be = vals.index(bv) + 1
                ax.axhline(bv, color=color, linestyle="--", alpha=0.5, label=f"Best {bv:.2f}% @ ep{be}")
                ax.set_title(label, fontsize=11); ax.set_xlabel("Epoch"); ax.set_ylabel("AP (%)")
                ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
            plt.tight_layout(); _savefig(fig, "03_ap_curves.png")

        # ── 04 PR 곡선 & F1/Confidence 분석 ──────────────────────────────
        if prec_arr is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f"PR Curve & Confidence Analysis — Epoch {epoch_num}", fontsize=12, fontweight="bold")

            # PR 곡선 (IoU=0.5, 0.75)
            ax = axes[0]
            for t_idx, iou_val, color in [(0, 0.50, "royalblue"), (5, 0.75, "tomato")]:
                p = prec_arr[t_idx, :, 0, 0, -1]
                valid = p > -1
                if valid.any():
                    ap_val = np.mean(p[valid]) * 100
                    ax.plot(r_thresh[valid], p[valid], color=color, linewidth=2,
                            label=f"IoU={iou_val:.2f}  AP={ap_val:.1f}%")
            ax.fill_between(r_thresh,
                            np.where(prec_arr[0, :, 0, 0, -1] > -1, prec_arr[0, :, 0, 0, -1], 0),
                            alpha=0.1, color="royalblue")
            ax.set_xlabel("Recall", fontsize=11); ax.set_ylabel("Precision", fontsize=11)
            ax.set_title("Precision-Recall Curve"); ax.legend(fontsize=10)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

            # P / R / F1 vs Confidence
            ax2 = axes[1]
            p50 = prec_arr[0, :, 0, 0, -1]
            s50 = scores_arr[0, :, 0, 0, -1] if scores_arr is not None else None
            valid = p50 > -1
            if valid.any() and s50 is not None:
                pv = p50[valid]; rv = r_thresh[valid]; sv = s50[valid]
                order = np.argsort(sv)[::-1]
                sv = sv[order]; pv = pv[order]; rv = rv[order]
                f1 = 2 * pv * rv / (pv + rv + 1e-8)
                ax2.plot(sv, pv, color="royalblue",  linewidth=1.8, label="Precision")
                ax2.plot(sv, rv, color="tomato",     linewidth=1.8, label="Recall")
                ax2.plot(sv, f1, color="darkorange", linewidth=2,   label="F1")
                best_i = int(np.argmax(f1))
                ax2.axvline(sv[best_i], color="gray", linestyle="--", alpha=0.7,
                            label=f"Best F1={f1[best_i]:.3f} @ conf={sv[best_i]:.3f}")
            ax2.set_xlabel("Confidence", fontsize=11); ax2.set_ylabel("Score", fontsize=11)
            ax2.set_title("Precision / Recall / F1 vs Confidence")
            ax2.legend(fontsize=9); ax2.set_xlim(0, 1); ax2.grid(True, alpha=0.3)
            plt.tight_layout(); _savefig(fig, "04_pr_f1_conf.png")

        # ── 05 IoU 분포 & Confidence 분포 ────────────────────────────────
        if output_data and coco_gt:
            all_ious = []; all_scores_flat = []
            tp = fp = fn = 0

            for img_id, pred in output_data.items():
                bboxes = pred.get("bboxes", [])
                sc     = pred.get("scores", [])
                all_scores_flat.extend(sc)
                ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
                gt_boxes = [
                    [a["bbox"][0], a["bbox"][1],
                     a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                    for a in coco_gt.loadAnns(ann_ids)
                ]
                if not gt_boxes:
                    fp += len(bboxes); continue
                if not bboxes:
                    fn += len(gt_boxes); continue
                matched_gt = [False] * len(gt_boxes)
                for box in bboxes:
                    ious = [self._box_iou_xyxy(box, g) for g in gt_boxes]
                    best_i = int(np.argmax(ious)); best_iou = ious[best_i]
                    all_ious.append(best_iou)
                    if best_iou >= 0.5 and not matched_gt[best_i]:
                        tp += 1; matched_gt[best_i] = True
                    else:
                        fp += 1
                fn += matched_gt.count(False)

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Detection Statistics — Epoch {epoch_num}", fontsize=12, fontweight="bold")

            ax = axes[0]
            if all_ious:
                ious_np = np.array(all_ious)
                ax.hist(ious_np[ious_np >= 0.5], bins=30, color="mediumseagreen", alpha=0.75,
                        label=f"TP (n={int((ious_np>=0.5).sum())})")
                ax.hist(ious_np[ious_np <  0.5], bins=30, color="tomato",         alpha=0.75,
                        label=f"FP (n={int((ious_np<0.5).sum())})")
                ax.axvline(0.5, color="gray", linestyle="--", linewidth=1.5)
                ax.set_xlabel("Max IoU with GT"); ax.set_ylabel("Count")
                ax.set_title("IoU Distribution Histogram"); ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

            ax2 = axes[1]
            if all_scores_flat:
                ax2.hist(all_scores_flat, bins=40, color="steelblue", alpha=0.8, edgecolor="white")
                ax2.set_xlabel("Confidence Score"); ax2.set_ylabel("Count")
                ax2.set_title("Confidence Score Distribution")
            ax2.grid(True, alpha=0.3)
            plt.tight_layout(); _savefig(fig, "05_iou_conf_dist.png")

            # ── 06 Confusion Matrix ───────────────────────────────────────
            prec_v  = tp / (tp + fp + 1e-8)
            rec_v   = tp / (tp + fn + 1e-8)
            f1_v    = 2 * prec_v * rec_v / (prec_v + rec_v + 1e-8)
            cm      = np.array([[tp, fp], [fn, 0]], dtype=float)
            cm_norm = cm / (cm.sum() + 1e-8)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(
                f"Confusion Matrix — Epoch {epoch_num}  |  P={prec_v:.3f}  R={rec_v:.3f}  F1={f1_v:.3f}",
                fontsize=12, fontweight="bold"
            )
            labels = [["TP", "FP"], ["FN", "TN*"]]
            for ax, data, title in [
                (axes[0], cm,      "Counts"),
                (axes[1], cm_norm, "Normalized"),
            ]:
                im = ax.imshow(data, cmap="Blues", vmin=0)
                plt.colorbar(im, ax=ax, fraction=0.046)
                thresh = data.max() / 2.0
                for i in range(2):
                    for j in range(2):
                        val = data[i, j]
                        txt = f"{int(val)}" if title == "Counts" else f"{val:.3f}"
                        ax.text(j, i, f"{labels[i][j]}\n{txt}", ha="center", va="center",
                                color="white" if val > thresh else "black", fontsize=11)
                ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
                ax.set_xticklabels(["Pred Pos", "Pred Neg"])
                ax.set_yticklabels(["GT Pos", "GT Neg"])
                ax.set_title(title, fontsize=11)
            axes[1].text(0, -0.12, "* TN = background (not countable in detection)",
                         transform=axes[1].transAxes, fontsize=8, color="gray")
            plt.tight_layout(); _savefig(fig, "06_confusion_matrix.png")

            # ── 07 Detection Samples (Best / Worst / FP / FN) ────────────
            self._save_detection_samples(fig_dir, output_data, coco_gt, dataset, epoch_num)

        # ── 08 메트릭 추세 & Overfitting 체크 ───────────────────────────
        if len(ap_epochs) >= 2:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.suptitle(f"Metric Trend & Overfitting Check — Epoch {epoch_num}", fontsize=12, fontweight="bold")

            ax = axes[0]
            vals95 = [v * 100 for v in self._ap_history["ap50_95"]]
            vals50 = [v * 100 for v in self._ap_history["ap50"]]
            window = min(10, max(1, len(ap_epochs) // 5))

            def _ma(arr, w):
                return [np.mean(arr[max(0, i-w+1):i+1]) for i in range(len(arr))]

            ax.plot(ap_epochs, vals95, color="royalblue",      alpha=0.35, linewidth=1)
            ax.plot(ap_epochs, vals50, color="mediumseagreen", alpha=0.35, linewidth=1)
            ax.plot(ap_epochs, _ma(vals95, window), color="royalblue",
                    linewidth=2, label=f"mAP@.5:.95 (MA{window})")
            ax.plot(ap_epochs, _ma(vals50, window), color="mediumseagreen",
                    linewidth=2, label=f"mAP@.50   (MA{window})")
            ax.set_xlabel("Epoch"); ax.set_ylabel("AP (%)")
            ax.set_title("AP Trend (Moving Average)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

            ax2 = axes[1]
            if self._loss_history["total_loss"]:
                ax2r = ax2.twinx()
                ax2.plot(epochs, self._loss_history["total_loss"],
                         color="tomato", linewidth=1.5, label="Train Loss")
                ax2r.plot(ap_epochs, vals95, color="royalblue",
                          linewidth=1.5, linestyle="--", label="Val AP50:95")
                ax2.set_xlabel("Epoch")
                ax2.set_ylabel("Train Loss", color="tomato")
                ax2r.set_ylabel("Val AP (%)", color="royalblue")
                ax2.tick_params(axis="y", labelcolor="tomato")
                ax2r.tick_params(axis="y", labelcolor="royalblue")
                ax2.set_title("Train Loss vs Val AP (Overfitting Check)")
                lines = ax2.get_legend_handles_labels()[0] + ax2r.get_legend_handles_labels()[0]
                lbls  = ax2.get_legend_handles_labels()[1] + ax2r.get_legend_handles_labels()[1]
                ax2.legend(lines, lbls, fontsize=9); ax2.grid(True, alpha=0.3)
            plt.tight_layout(); _savefig(fig, "07_metric_trends.png")

        logger.info(f"[Figures] Epoch {epoch_num} → {fig_dir}")

    def _save_detection_samples(self, fig_dir, output_data, coco_gt, dataset, epoch_num):
        """Best / Worst / FP / FN 검출 샘플 이미지 저장"""
        import numpy as np

        img_stats = []  # (img_id, mean_iou, max_conf, tp_n, fp_n, fn_n)
        for img_id, pred in output_data.items():
            bboxes = pred.get("bboxes", [])
            sc     = pred.get("scores", [])
            ann_ids = coco_gt.getAnnIds(imgIds=[img_id])
            gt_boxes = [
                [a["bbox"][0], a["bbox"][1],
                 a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                for a in coco_gt.loadAnns(ann_ids)
            ]
            if not (bboxes or gt_boxes):
                continue
            ious = []
            matched_gt = [False] * len(gt_boxes)
            tp_n = fp_n = 0
            for box in bboxes:
                if gt_boxes:
                    iou_list = [self._box_iou_xyxy(box, g) for g in gt_boxes]
                    bi = int(np.argmax(iou_list)); biou = iou_list[bi]
                    ious.append(biou)
                    if biou >= 0.5 and not matched_gt[bi]:
                        tp_n += 1; matched_gt[bi] = True
                    else:
                        fp_n += 1
                else:
                    fp_n += 1
            fn_n = matched_gt.count(False)
            mean_iou  = float(np.mean(ious)) if ious else 0.0
            max_conf  = max(sc) if sc else 0.0
            img_stats.append((img_id, mean_iou, max_conf, tp_n, fp_n, fn_n))

        if not img_stats:
            return

        def _draw(ax, img_id, title):
            info, img = self._load_img(img_id, coco_gt, dataset)
            if img is None:
                ax.axis("off"); ax.set_title(title, fontsize=7); return
            ax.imshow(img)
            # GT 박스 (초록)
            for ann in coco_gt.loadAnns(coco_gt.getAnnIds(imgIds=[img_id])):
                x, y, w, h = ann["bbox"]
                ax.add_patch(plt.Rectangle((x, y), w, h, linewidth=1.5,
                                           edgecolor="lime", facecolor="none"))
            # Pred 박스 (빨강)
            pred = output_data.get(img_id, {})
            for box, sc in zip(pred.get("bboxes", []), pred.get("scores", [])):
                x1, y1, x2, y2 = box
                ax.add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1.5,
                                           edgecolor="red", facecolor="none"))
                ax.text(x1, max(y1-3, 0), f"{sc:.2f}", fontsize=5,
                        color="red", va="bottom", clip_on=True)
            ax.axis("off"); ax.set_title(title, fontsize=7)

        N = 8  # 그룹당 샘플 수

        # Best: mean_iou 상위 (tp_n > 0 조건)
        best = sorted([s for s in img_stats if s[3] > 0], key=lambda x: -x[1])[:N]
        # Worst: 예측은 있으나 IoU 낮은 케이스
        worst = sorted([s for s in img_stats if s[0] in output_data and output_data[s[0]].get("bboxes")],
                       key=lambda x: x[1])[:N]
        # FP 많은 이미지
        fp_heavy = sorted(img_stats, key=lambda x: -x[4])[:N]
        # FN 많은 이미지
        fn_heavy = sorted(img_stats, key=lambda x: -x[5])[:N]

        for group_name, group, fname in [
            ("Best Detections (highest IoU)",          best,     "08_detection_best.png"),
            ("Worst Detections (lowest IoU with pred)","worst",  "09_detection_worst.png"),
            ("False Positive Heavy",                   fp_heavy, "10_detection_fp.png"),
            ("False Negative Heavy",                   fn_heavy, "11_detection_fn.png"),
        ]:
            # worst는 변수 이름 충돌 방지
            samples = worst if fname == "09_detection_worst.png" else group
            if not samples:
                continue
            cols = 4; rows = max(1, (len(samples) + cols - 1) // cols)
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
            fig.suptitle(
                f"{group_name} — Epoch {epoch_num}  |  Green=GT  Red=Pred",
                fontsize=11, fontweight="bold"
            )
            axes_flat = axes.flat if hasattr(axes, "flat") else [axes]
            for ax, (img_id, miou, mconf, tp_n, fp_n, fn_n) in zip(axes_flat, samples):
                _draw(ax, img_id, f"IoU={miou:.2f} conf={mconf:.2f} TP={tp_n} FP={fp_n} FN={fn_n}")
            for ax in list(axes_flat)[len(samples):]:
                ax.axis("off")
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, fname), dpi=100, bbox_inches="tight")
            plt.close(fig)

    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
                "curr_ap": ap,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(
                    self.file_name,
                    ckpt_name,
                    update_best_ckpt,
                    metadata={
                        "epoch": self.epoch + 1,
                        "optimizer": self.optimizer.state_dict(),
                        "best_ap": self.best_ap,
                        "curr_ap": ap
                    }
                )

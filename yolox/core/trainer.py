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
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
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

        with torch.cuda.amp.autocast(enabled=self.amp_training):
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
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

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

    def _save_training_figures(self):
        fig_dir = os.path.join(self.file_name, "figures")
        os.makedirs(fig_dir, exist_ok=True)

        epoch_num = self.epoch + 1
        epochs = list(range(1, epoch_num + 1))
        ap_epochs = list(range(1, len(self._ap_history["ap50_95"]) + 1))

        fig = plt.figure(figsize=(18, 12))
        fig.suptitle(
            f"{self.exp.exp_name}  —  Epoch {epoch_num}  |  Best AP50:95 = {self.best_ap * 100:.2f}%",
            fontsize=14, fontweight="bold"
        )
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

        # 1) Total Loss
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs, self._loss_history["total_loss"], color="steelblue", linewidth=1.5)
        ax1.set_title("Total Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)

        # 2) Component Losses
        ax2 = fig.add_subplot(gs[0, 1])
        for key, color in [("iou_loss", "tomato"), ("conf_loss", "darkorange"),
                           ("cls_loss", "mediumseagreen"), ("l1_loss", "mediumpurple")]:
            ax2.plot(epochs, self._loss_history[key], label=key, color=color, linewidth=1.2)
        ax2.set_title("Component Losses")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3) Learning Rate
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(epochs, self._lr_history, color="slateblue", linewidth=1.5)
        ax3.set_title("Learning Rate")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("LR")
        ax3.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax3.grid(True, alpha=0.3)

        # 4) AP50:95
        ax4 = fig.add_subplot(gs[1, 0])
        if ap_epochs:
            ax4.plot(ap_epochs, [v * 100 for v in self._ap_history["ap50_95"]],
                     color="royalblue", linewidth=1.5, marker=".")
            best_val = max(self._ap_history["ap50_95"]) * 100
            ax4.axhline(best_val, color="royalblue", linestyle="--", alpha=0.5,
                        label=f"Best {best_val:.2f}%")
            ax4.legend(fontsize=8)
        ax4.set_title("AP@[0.50:0.95]")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("AP (%)")
        ax4.grid(True, alpha=0.3)

        # 5) AP50
        ax5 = fig.add_subplot(gs[1, 1])
        if ap_epochs:
            ax5.plot(ap_epochs, [v * 100 for v in self._ap_history["ap50"]],
                     color="mediumseagreen", linewidth=1.5, marker=".")
            best_val50 = max(self._ap_history["ap50"]) * 100
            ax5.axhline(best_val50, color="mediumseagreen", linestyle="--", alpha=0.5,
                        label=f"Best {best_val50:.2f}%")
            ax5.legend(fontsize=8)
        ax5.set_title("AP@0.50")
        ax5.set_xlabel("Epoch")
        ax5.set_ylabel("AP (%)")
        ax5.grid(True, alpha=0.3)

        # 6) AP50 vs AP50:95 비교
        ax6 = fig.add_subplot(gs[1, 2])
        if ap_epochs:
            ax6.plot(ap_epochs, [v * 100 for v in self._ap_history["ap50_95"]],
                     color="royalblue", linewidth=1.5, label="AP50:95")
            ax6.plot(ap_epochs, [v * 100 for v in self._ap_history["ap50"]],
                     color="mediumseagreen", linewidth=1.5, label="AP50")
            ax6.legend(fontsize=8)
        ax6.set_title("AP Comparison")
        ax6.set_xlabel("Epoch")
        ax6.set_ylabel("AP (%)")
        ax6.grid(True, alpha=0.3)

        save_path = os.path.join(fig_dir, f"epoch_{epoch_num:04d}.png")
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Training figure saved → {save_path}")

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

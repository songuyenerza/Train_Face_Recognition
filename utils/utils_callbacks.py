import logging
import os
import time
from typing import List

import torch
from dataset import get_dataloader_eval

from eval import verification
from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed
from config import config as cfg
import torch.nn.functional as F


class CallBackVerification(object):
    
    def __init__(self, val_dir, local_rank, summary_writer=None, image_size=(112, 112), wandb_logger=None):
        self.rank: int = distributed.get_rank()
        self.highest_acc: float = 0.0
        self.local_rank = local_rank

        if local_rank == 0:
            self.init_dataset(val_dir=val_dir, image_size=image_size)

        self.summary_writer = summary_writer
        self.wandb_logger = wandb_logger

    def compute_accuracy(self, embeddings, labels):

        embeddings = F.normalize(embeddings, p=2, dim=1)
        similarity_matrix = torch.matmul(embeddings, embeddings.t())
        
        mask = torch.eye(similarity_matrix.size(0), device=similarity_matrix.device).bool()
        similarity_matrix.masked_fill_(mask, -1)

        # Tìm nearest neighbor
        max_similarities, nearest_indices = similarity_matrix.max(dim=1)
        nearest_labels = labels[nearest_indices]

        # Tính accuracy
        accuracy = (nearest_labels == labels).float().mean().item()
        return accuracy

    def evaluate(self, backbone: torch.nn.Module, global_step: int):
        backbone.eval()
        all_embeddings = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(backbone.device)
                labels = labels.to(backbone.device)
                embeddings = backbone(inputs)
                all_embeddings.append(embeddings)
                all_labels.append(labels)

        all_embeddings = torch.cat(all_embeddings)
        all_labels = torch.cat(all_labels)

        accuracy = self.compute_accuracy(all_embeddings, all_labels)
        print(f"Step {global_step}: Validation Accuracy: {accuracy * 100:.2f}%  ")

        if self.wandb_logger:
            import wandb
            self.wandb_logger.log({
                f'EVal/Val-Accuracy ': accuracy,
            })

        if accuracy > self.highest_acc:
            self.highest_acc = accuracy
        logging.info(
            '[%d]Accuracy-Highest: %1.5f' % (global_step, self.highest_acc))

    def init_dataset(self, val_dir, image_size):
        print("val_dir ::: ", val_dir)
        self.val_loader = get_dataloader_eval(
                        val_dir,
                        cfg.batch_size,
                        cfg.seed,
                        cfg.num_workers
                    )
        print("val_loader::: ", len(self.val_loader))
    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.local_rank == 0 and num_update > 0:
            backbone.eval()
            self.evaluate(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, start_step=0,writer=None):
        self.frequent: int = frequent
        self.rank: int = distributed.get_rank()
        self.world_size: int = distributed.get_world_size()
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step: int = start_step
        self.batch_size: int = batch_size
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(self,
                 global_step: int,
                 loss: AverageMeter,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                #time_now = (time.time() - self.time_start) / 3600
                #time_total = time_now / ((global_step + 1) / self.total_step)
                #time_for_end = time_total - time_now
                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (global_step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_step - global_step - 1)
                time_for_end = eta_sec/3600
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('learning_rate', learning_rate, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                if fp16:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step,
                              grad_scaler.get_scale(), time_for_end
                          )
                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step, time_for_end
                          )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()

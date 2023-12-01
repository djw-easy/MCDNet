import sys, math
import time, datetime
import numpy as np

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .loss import BoundaryLoss, mmIoULoss


class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.next_cloudy, self.next_dc, self.next_label = next(self.loader)
        except StopIteration:
            self.next_cloudy = None
            self.next_dc = None
            self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_cloudy = self.next_cloudy.cuda(non_blocking=True)
            self.next_dc = self.next_dc.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        cloudy, dc, label = self.next_cloudy, self.next_dc, self.next_label
        if cloudy is not None:
            cloudy.record_stream(torch.cuda.current_stream())
        if dc is not None:
            dc.record_stream(torch.cuda.current_stream())
        if label is not None:
            label.record_stream(torch.cuda.current_stream())
        self.preload()
        return cloudy, dc, label


class BaseTrainer(object):

    def __init__(self, args, model, device) -> None:
        self.model_name = args.model_name
        self.model = model
        self.device = device
        self.args = args
        self.cel = CrossEntropyLoss().to(device)
        self.boundary_loss = BoundaryLoss()
        self.mmiou_loss = mmIoULoss(n_classes=self.args.num_classes)
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer, self.lr_scheduler = None, None
        if self.model_name == 'cdnetv2':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-4, 
                                             momentum=0.9, weight_decay=0.0005)
            lr_decay_function = lambda epoch: (1 - epoch / self.args.n_epochs) ** 0.9
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_decay_function)
        elif self.model_name in ['cloudnet', 'unet', 'deeplabv3plus', 'segnet']:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, betas=(0.9, 0.999))
            lr_decay_function = lambda epoch: (1 - epoch / self.args.n_epochs) ** 0.9
            self.lr_scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_decay_function)
        elif self.model_name == 'mffsnet':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.0001)
            lr_decay_function = lambda epoch: (1 - epoch / self.args.n_epochs) ** 0.9
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_decay_function)
        elif self.model_name == 'dcnet':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-6)
            lr_decay_function = lambda epoch: (1 - epoch / self.args.n_epochs) ** 0.9
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_decay_function)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(self.args.b1, self.args.b2))
            lr_decay_function = lambda epoch: (1 - epoch / self.args.n_epochs) ** 0.9
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_decay_function)
        if self.args.start_epoch>1 and self.lr_scheduler!=None:
            self.lr_scheduler.step(self.args.start_epoch-1)

    def cal_loss(self, cloudy, dc, label):
        if self.model_name == 'cdnetv2':
            pred, pred_aux = self.model(cloudy)
            loss_pred = self.cel(pred, label)
            loss_aux = self.cel(pred_aux, label)
            loss = loss_pred + loss_aux
        elif self.model_name == 'mcdnet':
            predict = self.model(cloudy, dc)
            loss = self.cel(predict, label)
        else:
            predict = self.model(cloudy)
            loss = self.cel(predict, label)
        return loss

    def train(self, epoch, train_loader):
        prev_time = time.time()
        prefetcher = data_prefetcher(train_loader)
        cloudy, dc, label = prefetcher.next()
        i = 1
        while cloudy is not None:
            self.optimizer.zero_grad()
            loss = self.cal_loss(cloudy, dc, label)
            loss.backward()
            self.optimizer.step()

            # Determine approximate time left
            batches_done = (epoch - 1) * len(train_loader) + i
            batches_left = self.args.n_epochs * len(train_loader) - batches_done
            time_left = datetime.timedelta(seconds=int(batches_left * (time.time() - prev_time)))
            prev_time = time.time()
            
            #  Log Progress
            sys.stdout.write(
                "\r[Epoch %03d/%d] [Batch %03d/%d] [Cross Entropy Loss: %7.4f] ETA: %8s"
                % (
                    epoch,
                    self.args.n_epochs,
                    i,
                    len(train_loader),
                    loss.item(),
                    time_left,
                )
            )
            i += 1
            cloudy, dc, label = prefetcher.next()
        
        if self.lr_scheduler != None:
            self.lr_scheduler.step()




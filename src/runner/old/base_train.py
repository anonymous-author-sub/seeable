from __future__ import print_function, division

import sys
import time
import os
import csv
import random
import numpy as np

import torch
import torch.nn as nn

from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler

from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix

from src.transform import xception_transforms
from src.encoder import xception
from src.dataset import DatasetCSV
from src.utils import Logger


def make_weights_for_balanced_classes(train_dataset, stage="train"):
    targets = []

    targets = torch.tensor(train_dataset)

    class_sample_count = torch.tensor(
        [(targets == t).sum() for t in torch.unique(targets, sorted=True)]
    )
    weight = 1.0 / class_sample_count.float()
    samples_weight = torch.tensor([weight[t] for t in targets])
    return samples_weight

def base_data(csv_file):
    frame_reader = open(csv_file, "r")
    csv_reader = csv.reader(frame_reader)
    for f in csv_reader:
        path = f[0]
        label = int(f[1])
        train_label.append(label)
        train_list.append(path)
    self.log.write(str(len(train_list)) + "\n")


def validation_data(csv_file):
    frame_reader = open(csv_file, "r")
    fnames = csv.reader(frame_reader)
    for f in fnames:
        path = f[0]
        label = int(f[1])
        test_label.append(label)
        test_list.append(path)
    frame_reader.close()
    self.log.write(str(len(test_label)) + "\n")


class BaseTraining:

    def __init__(self, cfg):
        self.cfg = cfg

        # setup device
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        current_epoch = 0
        batch_size = 32
        train_csv = "/media/borutb/disk11/DataBase/DeepFake/Celeb-DF-v2/train/dfgc_train.csv"  # The train split file
        val_csv = "/media/borutb/disk11/DataBase/DeepFake/Celeb-DF-v2/train/dfgc_val.csv"  # The validation split file
        
        #  Output path
        model_dir = "/media/borutb/disk11/DataBase/DeepFake/Celeb-DF-v2/models/xception/"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        log_name = model_dir.split("/")[-2] + ".log"
        log_dir = os.path.join(model_dir, log_name)
        if os.path.exists(log_dir):
            # os.remove(log_dir)
            print("The log file exist!")

        log = Logger(log_dir, sys.stdout)
        self.log.write(
            f"model: xception | batch_size: {batch_size} | frames: I-frames \n"
        )
        self.log.write("pretrain:True | input_size:299*299\n")

        
        # Data loading parameters
        params = {}
        if self.use_cuda:
            params.update({"shuffle": False, "num_workers": 4, "pin_memory": True})
        

        train_list = []
        train_label = []
        self.log.write("loading train data" + "\n")
        base_data(train_csv)

        ziplist = list(zip(train_list, train_label))
        random.shuffle(ziplist)
        train_list[:], train_label[:] = zip(*ziplist)

        test_list = []
        test_label = []

        self.log.write("loading val data" + "\n")
        validation_data(val_csv)

        train_set, valid_set = (
            DatasetCSV(train_list, train_label, transform=xception_transforms),
            DatasetCSV(test_list, test_label, transform=xception_transforms),
        )

        images_datasets = {}
        images_datasets["train"] = train_label
        images_datasets["test"] = test_label

        weights = {
            x: make_weights_for_balanced_classes(images_datasets[x], stage=x)
            for x in ["train", "test"]
        }
        data_sampler = {
            x: WeightedRandomSampler(weights[x], len(images_datasets[x]), replacement=True)
            for x in ["train", "test"]
        }

        image_datasets = {}
        # over sampling
        image_datasets["train"] = DataLoader(
            train_set, sampler=data_sampler["train"], batch_size=batch_size, **params
        )
        # image_datasets['train'] = data.DataLoader(train_set, batch_size=batch_size, **params)
        image_datasets["test"] = DataLoader(valid_set, batch_size=batch_size, **params)

        dataloaders = {x: image_datasets[x] for x in ["train", "test"]}
        datasets_sizes = {x: len(image_datasets[x]) for x in ["train", "test"]}

    def get_model(self, cfg):
        self.model = xception(pretrained=True)
        self.model.train()
        self.model = nn.DataParallel(self.model.cuda())
        return self.model

        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion.cuda()

        optimizer_ft = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.001)
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=2, gamma=0.5)


    def train(self, epoch):
        best_logloss = 10.0
        best_epoch = 0
        for epoch in range(current_epoch, num_epochs):
            best_test_logloss = 10.0
            epoch_start = time.time()
            model_out_path = os.path.join(model_dir, str(epoch) + "_xception.ckpt")
            self.log.write(
                "------------------------------------------------------------------------\n"
            )
            # Each epoch has a training and validation phase
            for phase in ["train", "test"]:
                if phase == "train":
                    # self.scheduler.step() #add BB: Warning: you should call `self.optimizer.step()` before `lr_scheduler.step()`.
                    self.model.train()
                else:
                    self.model.eval()
                running_loss = 0.0
                running_loss_train = 0.0

                y_scores, y_trues = [], []
                for i, (inputs, labels) in enumerate(dataloaders[phase]):
                    inputs, labels = inputs.cuda(), labels.to(torch.float32).cuda()

                    if phase == "train":
                        self.optimizer.zero_grad()
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        preds = torch.sigmoid(outputs)
                        loss.backward()
                        self.optimizer.step()
                    else:
                        with torch.no_grad():
                            outputs = self.model(inputs)
                            # labels = labels.unsqueeze(1)
                            loss = self.criterion(outputs, labels)
                            preds = torch.sigmoid(outputs)
                    batch_loss = loss.data.item()
                    running_loss += batch_loss
                    running_loss_train += batch_loss

                    y_true = labels.data.cpu().numpy()
                    y_score = preds.data.cpu().numpy()

                    if i % 100 == 0:
                        batch_acc = accuracy_score(y_true, np.where(y_score > 0.5, 1, 0))
                        self.log.write(
                            "Epoch {}/{} Batch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n".format(
                                epoch,
                                num_epochs - 1,
                                i,
                                len(dataloaders[phase]),
                                phase,
                                batch_loss,
                                batch_acc,
                            )
                        )
                    if (i + 1) % 500 == 0:
                        inter_loss = running_loss_train / 500.0
                        self.log.write("last phase train loss is {}\n".format(inter_loss))
                        running_loss_train = 0.0
                        test_loss = self.validate(
                            self.model, self.criterion, num_epochs, test_list, epoch
                        )
                        if test_loss < best_test_logloss:
                            best_test_logloss = test_loss
                            self.log.write(
                                "save current self.model {}, Now time is {}, best logloss is {}\n".format(
                                    i,
                                    time.asctime(time.localtime(time.time())),
                                    best_test_logloss,
                                )
                            )
                            model_out_paths = os.path.join(
                                model_dir, str(epoch) + str(i) + "_xception.ckpt"
                            )
                            torch.save(self.model.module.state_dict(), model_out_paths)
                        self.model.train()
                        # self.scheduler.step()
                        self.log.write("now lr is : {}\n".format(self.scheduler.get_lr()))

                    if phase == "test":
                        y_scores.extend(y_score)
                        y_trues.extend(y_true)
                if phase == "train":  # add BB
                    self.scheduler.step()
                if phase == "test":
                    epoch_loss = running_loss / (len(test_list) / batch_size)
                    y_trues, y_scores = np.array(y_trues), np.array(y_scores)
                    accuracy = accuracy_score(y_trues, np.where(y_scores > 0.5, 1, 0))

                    self.log.write(
                        "**Epoch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n".format(
                            epoch, num_epochs - 1, phase, epoch_loss, accuracy
                        )
                    )
                if phase == "test" and epoch_loss < best_logloss:
                    best_logloss = epoch_loss
                    best_epoch = epoch
                    torch.save(self.model.module.state_dict(), model_out_path)

            self.log.write(
                "Epoch {}/{} Time {}s\n".format(
                    epoch, num_epochs - 1, time.time() - epoch_start
                )
            )
        self.log.write("***************************************************")
        self.log.write(
            f"Best logloss {best_logloss:.4f} and Best Epoch is {best_epoch}\n"
        )


    def validate(self, self.model, self.criterion, num_epochs, test_list, current_epoch=0, phase="test"):
        self.log.write(
            "------------------------------------------------------------------------\n"
        )
        # Each epoch has a training and validation phase
        self.model.eval()
        running_loss_val = 0.0
        # print(phase)
        y_scores, y_trues = [], []
        for k, (inputs_val, labels_val) in enumerate(dataloaders[phase]):
            inputs_val, labels_val = inputs_val.cuda(), labels_val.to(torch.float32).cuda()
            with torch.no_grad():
                outputs_val = self.model(inputs_val)
                # labels = labels.unsqueeze(1)
                loss = self.criterion(outputs_val, labels_val)
                preds = torch.sigmoid(outputs_val)
            batch_loss = loss.data.item()
            running_loss_val += batch_loss

            y_true = labels_val.data.cpu().numpy()
            y_score = preds.data.cpu().numpy()

            if k % 100 == 0:
                batch_acc = accuracy_score(y_true, np.where(y_score > 0.5, 1, 0))
                self.log.write(
                    "Epoch {}/{} Batch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n".format(
                        current_epoch,
                        num_epochs - 1,
                        k,
                        len(dataloaders[phase]),
                        phase,
                        batch_loss,
                        batch_acc,
                    )
                )
            y_scores.extend(y_score)
            y_trues.extend(y_true)

        epoch_loss = running_loss_val / (len(test_list) / batch_size)
        y_trues, y_scores = np.array(y_trues), np.array(y_scores)
        accuracy = accuracy_score(y_trues, np.where(y_scores > 0.5, 1, 0))
        # model_out_paths = os.path.join(model_dir, str(current_epoch) + '_xception.ckpt')
        # torch.save(self.model.module.state_dict(), model_out_paths)
        self.log.write(
            "**Epoch {}/{} Stage: {} Logloss: {:.4f} Accuracy: {:.4f}\n".format(
                current_epoch, num_epochs - 1, phase, epoch_loss, accuracy
            )
        )
        tn, fp, fn, tp = confusion_matrix(y_trues, np.where(y_scores > 0.5, 1, 0)).ravel()
        self.log.write(
            "**Epoch {}/{} Stage: {} TNR: {:.2f} FPR: {:.2f} FNR: {:.2f} TPR: {:.2f} \n".format(
                current_epoch,
                num_epochs - 1,
                phase,
                tn / (fp + tn),
                fp / (fp + tn),
                fn / (tp + fn),
                tp / (tp + fn),
            )
        )
        self.log.write("***************************************************\n")
        # self.model.train()
        return epoch_loss


    def run(self):
        start = time.time()

        self.train(
            self.model=self.model,
            model_dir=model_dir,
            self.criterion=self.criterion,
            self.optimizer=optimizer_ft,
            self.scheduler=exp_lr_scheduler,
            num_epochs=5,
            current_epoch=current_epoch,
        )

        elapsed = time.time() - start
        self.self.log.write("Total time is {}.\n".format(elapsed))



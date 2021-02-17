import time

import torch
import os
from utils import *
from torch.autograd import Variable


class KdDistill:

    def __init__(self,
                 teacher_model,
                 student_model,
                 train_loader,
                 val_loader,
                 learning_rate,
                 decay_epoch,
                 alpha,
                 temperature,
                 device,
                 teacher_model_name,
                 student_model_name,
                 mode,
                 optimizer):

        self.teacher = teacher_model
        self.student = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = learning_rate
        self.decay_epoch = decay_epoch
        self.device = device
        self.teacher.to(self.device)
        self.student.to(self.device)

        self.alpha = alpha
        self.temperature = temperature

        self.teacher_model_name = teacher_model_name
        self.student_model_name = student_model_name
        self.mode = mode
        self.optimizer = optimizer

    def run(self, epochs):

        if not os.path.exists(f'./models/{self.teacher_model_name}'):
            os.makedirs(f'./models/{self.teacher_model_name}')

        train_loss = []
        eval_loss = []

        train_accuracy = []
        eval_accuracy = []

        best_accuracy = 0

        print("Started training for %d epochs" % epochs)

        for epoch in range(1, epochs):

            top_loss_train, top_accuracy_train = self.train(epoch)
            top_loss_eval, top_accuracy_eval = self.eval(epoch)

            train_loss.append(top_loss_train)
            eval_loss.append(top_loss_eval)

            train_accuracy.append(top_accuracy_train)
            eval_accuracy.append(top_accuracy_eval)

            if top_accuracy_eval > best_accuracy:
                best_accuracy = top_accuracy_eval
                if self.mode == 'distill':
                    torch.save(self.student.state_dict(),
                               f'./models/{self.teacher_model_name}/best_weights_{self.student_model_name}')
                else:
                    torch.save(self.teacher.state_dict(),
                               f'./models/{self.teacher_model_name}/best_weights_{self.teacher_model_name}')

    def train(self, epoch):

        if self.mode == 'distill':
            self.student.train()

        self.teacher.train()
        top_loss = AverageMeter()
        top_accuracy = AverageMeter()

        batch_size = len(self.train_loader)
        t1 = time.time()
        criterion = nn.CrossEntropyLoss().to(self.device)

        if epoch % self.decay_epoch == 0:
            self.adjust_lr()
            if epoch % self.decay_epoch * 2 == 0 and self.mode == 'distill':
                self.alpha = 0.2
                self.temperature = 1
                print(f'New alpha: {self.alpha} | New temperature: {self.temperature}')

        for i, (data, labels) in enumerate(self.train_loader):

            data, labels = data.to(self.device), labels.to(self.device)
            data, labels = Variable(data), Variable(labels)

            self.optimizer.zero_grad()

            batch_teacher_pred = self.teacher(data)

            if self.mode == 'distill':
                print('No')
                batch_student_pred = self.student(data)
                loss = self.kd_loss(batch_student_pred, batch_teacher_pred, labels)
                loss.backward()
                self.optimizer.step()
                train_acc = get_accuracy(batch_student_pred, labels)
                top_loss.update(loss.item(), data.size(0))
                top_accuracy.update(train_acc, data.size(0))

            else:
                loss = criterion(batch_teacher_pred, labels)
                loss.backward()
                self.optimizer.step()
                train_acc = get_accuracy(batch_teacher_pred, labels)
                top_loss.update(loss.item(), data.size(0))
                top_accuracy.update(train_acc, data.size(0))

            t2 = time.time()

            if (i % 100 == 0) or (i + 1 == batch_size) and not i == 0:
                print(f"{self.mode}: Epoch: {epoch} | "
                      f"Iteration: ({i}/{batch_size}) | "
                      f"Loss: {round(top_loss.avg, 2)} | "
                      f"Accuracy: {round(top_accuracy.avg, 2)} | "
                      f"Running: {round((t2 - t1), 2)} sec")

                top_loss = AverageMeter()
                top_accuracy = AverageMeter()

        return top_loss.avg, top_accuracy.avg

    def eval(self, epoch):

        model = None
        if self.mode == 'distill':
            model = self.student
        else:
            model = self.teacher

        model.eval()
        top_loss = AverageMeter()
        top_accuracy = AverageMeter()

        criterion = nn.CrossEntropyLoss().to(self.device)
        batch_size = len(self.val_loader)
        t1 = time.time()

        for i, (data, labels) in enumerate(self.val_loader):

            data, labels = data.to(self.device), labels.to(self.device)
            data, labels = Variable(data), Variable(labels)

            batch_pred = model(data)

            loss = criterion(batch_pred, labels)

            loss.backward()

            val_acc = get_accuracy(batch_pred, labels)
            top_loss.update(loss.item(), data.size(0))
            top_accuracy.update(val_acc, data.size(0))

            t2 = time.time()

            if ((i + 1) % 50 == 0) or i == batch_size:
                print(f"Validation: Epoch: {epoch} | "
                      f"Iteration: ({i}/{batch_size}) | "
                      f"Loss: {round(top_loss.avg, 2)} | "
                      f"Accuracy: {round(top_accuracy.avg, 2)} | "
                      f"Running: {round((t2 - t1), 2)} sec")

        return top_loss.avg, top_accuracy.avg

    def adjust_lr(self):

        self.learning_rate *= 0.8

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.learning_rate

        print(f"New Learning Rate from now on is {self.learning_rate}.")

    def kd_loss(self, student_pred, teacher_pred, labels):

        soft_target_loss = F.kl_div(F.log_softmax(student_pred / self.temperature, dim=1),
                                    F.softmax(teacher_pred / self.temperature, dim=1), reduction='batchmean')

        student_loss = F.cross_entropy(student_pred, labels)

        return soft_target_loss * (1 - self.alpha) + student_loss * self.alpha



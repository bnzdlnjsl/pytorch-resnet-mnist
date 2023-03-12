import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class Solver:
    def __init__(self, model, data_loaders: dict, optimizer, num_epochs: int, time_wait: int=10, device=None):
        self.model = model
        self.loader_train = data_loaders["train"]
        self.loader_val = data_loaders["val"]
        self.loader_test = data_loaders["test"]
        self.optimizer = optimizer
        self.num_epochs = num_epochs
        self.time_wait = time_wait
        self.device = device

    def check_accuracy(self, data_loader: DataLoader):
        self.model.eval()
        num_total = 0
        num_correct = 0
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                num_total += x.shape[0]
                out = self.model(x)
                _, y_pred = torch.max(out, 1)
                num_correct += torch.sum(y_pred == y)

        return num_correct / num_total

    def train(self):
        time_start = time.time()
        for epoch in range(self.num_epochs):
            self.model.train()
            for batch_idx, (x, y) in enumerate(self.loader_train):
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)
                loss = F.cross_entropy(out, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if batch_idx % 100 == 0:
                    print("epoch: {}/{}\tbatch: {}/{}\tloss: {:.6f}".format(
                        epoch + 1,
                        self.num_epochs,
                        batch_idx,
                        len(self.loader_train),
                        loss.item()
                    ))
            
            accuracy_val = self.check_accuracy(self.loader_val)
            print("val accuracy: {:.2f}%\ttime elapsed: {}".format(
                accuracy_val * 100,
                self.time_formatted(time.time() - time_start)
            ))
            time.sleep(self.time_wait)
            
    def time_formatted(self, seconds: float):
        temp = int(seconds)
        sec = temp % 60
        temp //= 60
        min = temp % 60
        temp //= 60
        hour = temp % 24
        return "{:02d}:{:02d}:{:02d}".format(hour, min, sec)
            
    def test(self):
        accuracy_test = self.check_accuracy(self.loader_test)
        print("test accuracy: {:.2f}%".format(accuracy_test * 100))
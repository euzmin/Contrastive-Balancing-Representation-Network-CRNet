
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class NN(nn.Module):

    def __init__(self, input_dim) -> None:
        super().__init__()
        self.layer_cfg = [200,200,128,50,50,1]

        self.net = []
        self.net.append(nn.Linear(input_dim, self.layer_cfg[0]))
        for i in range(1, len(self.layer_cfg)):
            self.net.append(nn.LeakyReLU())
            self.net.append(nn.Linear(self.layer_cfg[i-1], self.layer_cfg[i]))
        self.net = nn.Sequential(*self.net)
    def forward(self, x):
        return self.net(x)

    def criterion(self, out, y, w=1.0):
        loss = (w*(out.squeeze()-y.squeeze())**2).mean()
        return loss
    def predict(self, tx):
        out = self.forward(tx)
        return out.squeeze()
    def train_model(self, args, device, opt, train_data, val_data, adrf=None):
        self.train()
        train_data = torch.tensor(train_data, dtype=torch.float32)
        # val_data = torch.tensor(val_data, dtype=torch.float32)
        w = 1.0
        if train_data.shape[1] > args.t_dim + args.x_dim + 1:
            w = train_data[:,-1].to(device)
            train_data = train_data[:,:-1]

        train_loader = DataLoader(train_data, args.train_bs, shuffle=True)

        for epoch in range(100):
            for i, sample in enumerate(train_loader):
                
                opt.zero_grad()
                tx = sample[:, :-1].to(device)
                y = sample[:, -1].to(device)
                out = self(tx)
                loss = self.criterion(out, y, w)
                # opt.zero_grad()
                loss.backward()
                opt.step()
            if (epoch+1) % 100 == 0:
                mse = ((out.squeeze()-y.squeeze())**2).mean()
                print(epoch, i, loss.item(), mse)

        return self
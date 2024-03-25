import torch
import torch.nn as nn
import numpy as np
def comp_grid(y, num_grid):

    # L gives the lower index
    # U gives the upper index
    # inter gives the distance to the lower int
    # y 必须在0到1之间
    eps = 1e-5
    U = torch.ceil(y * num_grid)
    inter = 1 - (U - y * num_grid)
    L = U - 1
    L += (L < 0).int()

    return L.int().tolist(), U.int().tolist(), inter


class Density_Block(nn.Module):
    def __init__(self, num_grid, ind, isbias=1):
        super(Density_Block, self).__init__()
        """
        Assume the variable is bounded by [0,1]
        the output grid: 0, 1/B, 2/B, ..., B/B; output dim = B + 1; num_grid = B
        """
        self.ind = ind
        self.num_grid = num_grid
        self.outd = num_grid + 1

        self.isbias = isbias

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)
        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        self.softmax = nn.Softmax(dim=1)

    def forward(self, t, x):
        out = torch.matmul(x, self.weight)
        if self.isbias:
            out += self.bias
        out = self.softmax(out)

        x1 = list(torch.arange(0, x.shape[0]))
        L, U, inter = comp_grid(t, self.num_grid)

        L_out = out[x1, L]
        U_out = out[x1, U]

        out = L_out + (U_out - L_out) * inter

        return out


class Treat_Linear(nn.Module):
    def __init__(self, ind, outd, act='relu', istreat=1, isbias=1, islastlayer=0):
        super(Treat_Linear, self).__init__()
        # ind does NOT include the extra concat treatment
        self.ind = ind
        self.outd = outd
        self.isbias = isbias
        self.istreat = istreat
        self.islastlayer = islastlayer

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd), requires_grad=True)
        else:
            self.bias = None

        if self.istreat:
            self.treat_weight = nn.Parameter(torch.rand(1, self.outd), requires_grad=True)
        else:
            self.treat_weight = None

        if act == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = None

    def forward(self, x):
        # x: batch_size * (treatment, other feature)
        x_feature = x[:, 1:]
        x_treat = x[:, [0]]

        out = torch.matmul(x_feature, self.weight)

        if self.istreat:
            out = out + torch.matmul(x_treat, self.treat_weight)
        if self.isbias:
            out = out + self.bias

        if self.act is not None:
            out = self.act(out)

        if not self.islastlayer:
            out = torch.cat((x_treat, out), 1)

        return out

class Multi_head(nn.Module):
    def __init__(self, cfg, isenhance):
        super(Multi_head, self).__init__()

        self.cfg = cfg # cfg does NOT include the extra dimension of concat treatment
        self.isenhance = isenhance  # set 1 to concat treatment every layer/ 0: only concat on first layer

        # we default set num of heads = 5
        self.pt = [0.0, 0.2, 0.4, 0.6, 0.8, 1.]

        self.outdim = -1
        # construct the predicting networks
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                self.outdim = layer_cfg[1]
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                           islastlayer=0))
        blocks.append(last_layer)
        self.Q1 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q2 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q3 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q4 = nn.Sequential(*blocks)

        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg) - 1:  # last layer
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                last_layer = Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat,
                                          isbias=layer_cfg[2],
                                          islastlayer=1)
            else:
                if layer_idx == 0 or self.isenhance:
                    istreat = 1
                else:
                    istreat = 0
                blocks.append(
                    Treat_Linear(layer_cfg[0], layer_cfg[1], act=layer_cfg[3], istreat=istreat, isbias=layer_cfg[2],
                                 islastlayer=0))
        blocks.append(last_layer)
        self.Q5 = nn.Sequential(*blocks)

    def forward(self, x):
        # x = [treatment, features]
        out = torch.zeros(x.shape[0], self.outdim)
        t = x[:, 0]

        idx1 = list(set(list(torch.where(t >= self.pt[0])[0].numpy())) & set(torch.where(t < self.pt[1])[0].numpy()))
        idx2 = list(set(list(torch.where(t >= self.pt[1])[0].numpy())) & set(torch.where(t < self.pt[2])[0].numpy()))
        idx3 = list(set(list(torch.where(t >= self.pt[2])[0].numpy())) & set(torch.where(t < self.pt[3])[0].numpy()))
        idx4 = list(set(list(torch.where(t >= self.pt[3])[0].numpy())) & set(torch.where(t < self.pt[4])[0].numpy()))
        idx5 = list(set(list(torch.where(t >= self.pt[4])[0].numpy())) & set(torch.where(t <= self.pt[5])[0].numpy()))

        if idx1:
            out1 = self.Q1(x[idx1, :])
            out[idx1, :] = out[idx1, :] + out1

        if idx2:
            out2 = self.Q2(x[idx2, :])
            out[idx2, :] = out[idx2, :] + out2

        if idx3:
            out3 = self.Q3(x[idx3, :])
            out[idx3, :] = out[idx3, :] + out3

        if idx4:
            out4 = self.Q4(x[idx4, :])
            out[idx4, :] = out[idx4, :] + out4

        if idx5:
            out5 = self.Q5(x[idx5, :])
            out[idx5, :] = out[idx5, :] + out5

        return out



class Drnet(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, isenhance):
        super(Drnet, self).__init__()

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]

        self.cfg_density = cfg_density
        self.num_grid = num_grid
        self.cfg = cfg
        self.isenhance = isenhance

        # construct the density estimator
        density_blocks = []
        density_hidden_dim = -1
        for layer_idx, layer_cfg in enumerate(cfg_density):
            # fc layer
            if layer_idx == 0:
                # weight connected to feature
                self.feature_weight = nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2])
                density_blocks.append(self.feature_weight)
            else:
                density_blocks.append(nn.Linear(in_features=layer_cfg[0], out_features=layer_cfg[1], bias=layer_cfg[2]))
            density_hidden_dim = layer_cfg[1]
            if layer_cfg[3] == 'relu':
                density_blocks.append(nn.ReLU(inplace=True))
            elif layer_cfg[3] == 'tanh':
                density_blocks.append(nn.Tanh())
            elif layer_cfg[3] == 'sigmoid':
                density_blocks.append(nn.Sigmoid())
            else:
                print('No activation')

        self.hidden_features = nn.Sequential(*density_blocks)

        self.density_hidden_dim = density_hidden_dim
        self.density_estimator_head = Density_Block(self.num_grid, density_hidden_dim, isbias=1)

        # multi-head outputs blocks
        self.Q = Multi_head(cfg, isenhance)

    def forward(self, t, x):
        t = t.squeeze()
        hidden = self.hidden_features(x)
        t_hidden = torch.cat((torch.unsqueeze(t, 1), hidden), 1)
        # g = self.density_estimator_head(t, hidden)
        g = None
        Q = self.Q(t_hidden)

        return g, Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Treat_Linear):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()
                if m.istreat:
                    m.treat_weight.data.normal_(0, 1.)  # this needs to be initialized large to have better performance
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()

    def predict(self, tx):
        t = torch.tensor(tx[:, :1], dtype=torch.float32)
        x = torch.tensor(tx[:, 1:], dtype=torch.float32)
        out = self.forward(t.squeeze(), x)
        out = out[1].data.squeeze()
        return out
    def train_model(self, opt, train_data):
        t = train_data[:, :1]
        x = train_data[:, 1:-1]
        y = train_data[:, -1].reshape(-1,1)
        for epoch in range(100):
            opt.zero_grad()
            t = torch.tensor(t, dtype=torch.float32)
            x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.float32)
            out = self.forward(t.squeeze(), x)
            loss = self.criterion(out, y)
            if (epoch+1) % 100 == 0:
                print(f'{epoch} : {loss.mean()}')
            loss.backward()
            opt.step()
        return self
    def test_model(model, args, test_data, adrf):
        test_data = test_data
        adrf = adrf
        t_dim = args.t_dim
        tx_test = test_data[:, :-1]
        y_test = test_data[:, -1].reshape((-1, 1))
        out = model.predict(tx_test)
        mse = ((out.squeeze() - y_test.squeeze()) ** 2).mean()

        adrf_hat = np.zeros((adrf.shape[0]))
        for test_id in range(adrf.shape[0]):
            t_tmp = adrf[test_id, :t_dim].reshape((-1, t_dim)).repeat(test_data.shape[0], axis=0)
            tx_tmp = test_data[:, :-1]
            tx_tmp[:, :t_dim] = t_tmp
            out = model.predict(tx_tmp)
            y_hat = out.squeeze().mean()
            adrf_hat[test_id] = y_hat
        return adrf_hat
    def criterion(self, out, y, alpha=0.5, epsilon=1e-6):
        # return ((out[1].squeeze() - y.squeeze())**2).mean() - alpha * torch.log(out[0] + epsilon).mean()
        return ((out[1].squeeze() - y.squeeze())**2).mean()




"""
cfg_density = [(3,4,1,'relu'), (4,6,1,'relu')]
cfg = [(6,4,1,'relu'), (4,1,1,'id')]
num_grid = 10
isenhance = 1
D = Drnet(cfg_density, num_grid, cfg, isenhance)
D._initialize_weights()
x = torch.rand(10, 3)
t = torch.rand(10)
y = torch.rand(10)
out = D.forward(t, x)
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class Truncated_power():
    def __init__(self, degree, knots):
        """
        This class construct the truncated power basis; the data is assumed in [0,1]
        :param degree: int, the degree of truncated basis
        :param knots: list, the knots of the spline basis; two end points (0,1) should not be included
        """
        self.degree = degree
        self.knots = knots
        self.num_of_basis = self.degree + 1 + len(self.knots)
        self.relu = nn.ReLU(inplace=True)

        if self.degree == 0:
            print('Degree should not set to be 0!')
            raise ValueError

        if not isinstance(self.degree, int):
            print('Degree should be int')
            raise ValueError

    def forward(self, x):
        """
        :param x: torch.tensor, batch_size * 1
        :return: the value of each basis given x; batch_size * self.num_of_basis
        """
        x = x.squeeze()
        out = torch.zeros(x.shape[0], self.num_of_basis)
        for _ in range(self.num_of_basis):
            if _ <= self.degree:
                if _ == 0:
                    out[:, _] = 1.
                else:
                    out[:, _] = x**_
            else:
                if self.degree == 1:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree]))
                else:
                    out[:, _] = (self.relu(x - self.knots[_ - self.degree - 1])) ** self.degree

        return out # bs, num_of_basis


class Dynamic_FC(nn.Module):
    def __init__(self, ind, outd, degree, knots, act='relu', isbias=1, islastlayer=0):
        super(Dynamic_FC, self).__init__()
        self.ind = ind
        self.outd = outd
        self.degree = degree
        self.knots = knots

        self.islastlayer = islastlayer

        self.isbias = isbias

        self.spb = Truncated_power(degree, knots)
        self.d = self.spb.num_of_basis # num of basis

        self.weight = nn.Parameter(torch.rand(self.ind, self.outd, self.d), requires_grad=True)

        if self.isbias:
            self.bias = nn.Parameter(torch.rand(self.outd, self.d), requires_grad=True)
        else:
            self.bias = None

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
        x_treat = x[:, 0]

        x_feature_weight = torch.matmul(self.weight.T, x_feature.T).T # bs, outd, d

        x_treat_basis = self.spb.forward(x_treat) # bs, d
        x_treat_basis_ = torch.unsqueeze(x_treat_basis, 1)

        # x_feature_weight * x_treat_basis; bs, outd, d
        out = torch.sum(x_feature_weight * x_treat_basis_, dim=2) # bs, outd

        if self.isbias:
            out_bias = torch.matmul(self.bias, x_treat_basis.T).T
            out = out + out_bias

        if self.act is not None:
            out = self.act(out)

        # concat the treatment for intermediate layer
        if not self.islastlayer:
            out = torch.cat((torch.unsqueeze(x_treat, 1), out), 1)

        return out


def comp_grid(y, num_grid):

    # L gives the lower index
    # U gives the upper index
    # inter gives the distance to the lower int

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

class Vcnet(nn.Module):
    def __init__(self, cfg_density, num_grid, cfg, degree, knots):
        super(Vcnet, self).__init__()
        """
        cfg_density: cfg for the density estimator; [(ind1, outd1, isbias1), 'act', ....]; the cfg for density estimator head is not included
        num_grid: how many grid used for the density estimator head
        """

        # cfg/cfg_density = [(ind1, outd1, isbias1, activation),....]
        self.cfg_density = cfg_density
        self.num_grid = num_grid

        self.cfg = cfg
        self.degree = degree
        self.knots = knots

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

        # construct the dynamics network
        blocks = []
        for layer_idx, layer_cfg in enumerate(cfg):
            if layer_idx == len(cfg)-1: # last layer
                last_layer = Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=1)
            else:
                blocks.append(
                    Dynamic_FC(layer_cfg[0], layer_cfg[1], self.degree, self.knots, act=layer_cfg[3], isbias=layer_cfg[2], islastlayer=0))
        blocks.append(last_layer)

        self.Q = nn.Sequential(*blocks)
        self.linear1 = nn.Linear(in_features=201, out_features=1)
    def forward(self, t, x):
        hidden = self.hidden_features(x)
        t_hidden = torch.cat((torch.unsqueeze(t, 1), hidden), 1)
        #t_hidden = torch.cat((torch.unsqueeze(t, 1), x), 1)
        # g = self.density_estimator_head(t, hidden)
        g = None
        # Q = self.Q(t_hidden)
        Q = self.linear1(t_hidden)

        return g, Q

    def _initialize_weights(self):
        # TODO: maybe add more distribution for initialization
        for m in self.modules():
            if isinstance(m, Dynamic_FC):
                m.weight.data.normal_(0, 1.)
                if m.isbias:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, Density_Block):
                m.weight.data.normal_(0, 0.01)
                if m.isbias:
                    m.bias.data.zero_()

    def predict(self, tx):
        t = torch.tensor(tx[:, 0], dtype=torch.float32)
        x = torch.tensor(tx[:, 1:], dtype=torch.float32)
        out = self.forward(t.squeeze(), x)
        out = out[1].data.squeeze()
        return out

    def criterion(self, out, y, alpha=0.5, epsilon=1e-6):
        return ((out[1].squeeze() - y.squeeze())**2).mean()
        # return ((out[1].squeeze() - y.squeeze())**2).mean() - alpha * torch.log(out[0] + epsilon).mean()


    def train_model(self, opt, train_data):

        train_data = torch.tensor(train_data, dtype=torch.float32)
        train_loader = DataLoader(train_data, 3000, shuffle=True)
        for epoch in range(800):

            for idx, sample in enumerate(train_loader):
                inputs = sample[:, :-1]
                y = sample[:, -1]
                t = inputs[:, 0]
                x = inputs[:, 1:]

                opt.zero_grad()
                # out = [g, Q], g估计f(t|x), Q估计y
                out = self.forward(t, x)
                loss = self.criterion(out, y)
                
                if epoch % 100 == 0:
                    print(f'epoch:{epoch}, loss:{((out[1].squeeze() - y.squeeze())**2).mean()}')
                # loss = ((out[1].squeeze() - y.squeeze())**2).mean()
                loss.backward()
                opt.step()

                # _, y_out = out
                # y_loss = ((y_out - y) ** 2).mean()
        # t = train_data[:, :1]
        # x = train_data[:, 1:-1]
        # y = train_data[:, -1].reshape(-1,1)
        # for epoch in range(800):
        #     opt.zero_grad()
        #     t = torch.tensor(t, dtype=torch.float32)
        #     x = torch.tensor(x, dtype=torch.float32)
        #     y = torch.tensor(y, dtype=torch.float32)
        #     out = model.forward(t.squeeze(), x)
        #     loss = criterion(out, y)
        #     print(f'{epoch} : {loss.mean().data}')
        #     loss.backward()
        #     opt.step()
        return self
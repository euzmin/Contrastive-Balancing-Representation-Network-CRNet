import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy.random as random
from tqdm import tqdm
def U_distance_matrix(latent):
    n = latent.shape[0]
    matrix_a = torch.sqrt(torch.sum(torch.square(latent.unsqueeze(0) - latent.unsqueeze(1)), dim = -1)  + 1e-18)
    matrix_A = matrix_a - torch.sum(matrix_a, dim = 0, keepdims= True)/(n-2) - torch.sum(matrix_a, dim = 1, keepdims= True)/(n-2) \
                + torch.sum(matrix_a)/((n-1)*(n-2))

    diag_A = torch.diag(torch.diag(matrix_A) ) 
    matrix_A = matrix_A - diag_A
    return matrix_A


def U_product(matrix_A, matrix_B):
    n = matrix_A.shape[0]
    return torch.sum(matrix_A * matrix_B)/(n*(n-3))


def P_removal(matrix_A, matrix_C):
    result = matrix_A - U_product(matrix_A, matrix_C) / U_product(matrix_C, matrix_C) * matrix_C
    return result

def Correlation(matrix_A, matrix_B):
    Gamma_XY = U_product(matrix_A, matrix_B)
    Gamma_XX = U_product(matrix_A, matrix_A)
    Gamma_YY = U_product(matrix_B, matrix_B)

    # correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY + 1e-18)
    correlation_r = Gamma_XY/torch.sqrt(Gamma_XX * Gamma_YY)

    return correlation_r


def P_DC(latent_A, latent_B, ground_truth):
    matrix_A = U_distance_matrix(latent_A)
    matrix_B = U_distance_matrix(latent_B)
    matrix_GT = U_distance_matrix(ground_truth)

    matrix_A_B = P_removal(matrix_A, matrix_B)

    cr = Correlation(matrix_A_B, matrix_GT)

    return cr
def Partial_DC(x, y, z):
    a = U_distance_matrix(x)
    b = U_distance_matrix(y)
    c = U_distance_matrix(z)
    aa = U_product(a, a)
    bb = U_product(b, b)
    cc = U_product(c, c)
    ab = U_product(a, b)
    ac = U_product(a, c)
    bc = U_product(b, c)

    denom_sqr = aa * bb

    r_xy = ab / torch.sqrt(denom_sqr) if denom_sqr != 0 else denom_sqr

    denom_sqr = aa * cc
    r_xz = ac / torch.sqrt(denom_sqr) if denom_sqr != 0 else denom_sqr
    # r_xz = np.clip(r_xz, -1, 1)

    denom_sqr = bb * cc
    r_yz = bc / torch.sqrt(denom_sqr) if denom_sqr != 0 else denom_sqr
    # r_yz = np.clip(r_yz, -1, 1)

    denom = torch.sqrt(1.0 - r_xz ** 2+ 1e-18) * torch.sqrt(1.0 - r_yz ** 2+ 1e-18)
    p_dcor = (r_xy - r_xz * r_yz) / denom if denom != 0 else denom
    return p_dcor



class T_module(nn.Module):
    def __init__(self, t_dim) -> None:
        super().__init__()
        self.T_linear1 = nn.Linear(t_dim, 100)
        # self.T_linear2 = nn.Linear(100, 128)
        self.T_linear2 = nn.Linear(100, 32)

        # self.T_linear1 = nn.Linear(t_dim, 10)
        # self.T_linear2 = nn.Linear(10, 128)
        # self.T_linear3 = nn.Linear(64, 64)
        # self.T_linear4 = nn.Linear(64, 32)
        # self.T_linear5 = nn.Linear(32, 32)

    def forward(self, t):
        t_out1 = F.leaky_relu(self.T_linear1(t))
        t_out = F.leaky_relu(self.T_linear2(t_out1))
        # t_out = F.leaky_relu(self.T_linear3(t_out))
        # t_out = F.leaky_relu(self.T_linear4(t_out))
        # t_out = F.leaky_relu(self.T_linear5(t_out))
        return t_out

class X_module(nn.Module):
    def __init__(self, x_dim) -> None:
        super().__init__()
        self.X_linear1 = nn.Linear(x_dim, 100)
        # self.X_linear2 = nn.Linear(100, 128)
        self.X_linear2 = nn.Linear(100, 32)
        # self.X_linear3 = nn.Linear(50, 25)

    
    def forward(self, x):
        x_out = F.leaky_relu(self.X_linear1(x))
        out = F.leaky_relu(self.X_linear2(x_out))
        # out = F.leaky_relu(self.X_linear3(out))
        # out = F.leaky_relu(self.X_linear4(out))
        # out = F.leaky_relu(self.X_linear5(out))

        return out

class Backbone_module(nn.Module):
    def __init__(self, t_dim, x_dim, size) -> None:
        super().__init__()
        self.t_dim = t_dim
        self.x_module = X_module(x_dim)
        self.t_module = T_module(t_dim)
        # self.X_linear3 = nn.Linear(32,32)
        self.size = size
        # self.Linear1 = nn.Linear(128, 16)




    def pd_forward(self, t, x):
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        t_tmp = self.t_module.forward(t)
        x_tmp = self.x_module.forward(x)
        
        return t_tmp, x_tmp
    def forward(self, t, x):

        t_tmp, x_tmp = self.pd_forward(t,x)

        # representation = torch.cat((t_tmp, x_tmp), dim=0)
        # representation = F.leaky_relu(self.Linear1(representation))
        # out = torch.cat((t_tmp, x_tmp), dim=1)
        # representation = F.leaky_relu(self.Linear1(out))

        
        return t_tmp, x_tmp


class PT_Module(nn.Module):
    def __init__(self,z_dim) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.Linear1 = nn.Linear(self.z_dim*2, 50)

        self.Linear4 = nn.Linear(50, 1)
    def forward(self, inputs):
        out = F.leaky_relu(self.Linear1(inputs))
        out = self.Linear4(out)
        return out

class COB(nn.Module):

    def __init__(self, t_dim, x_dim, size) -> None:
        super().__init__()
        self.z_dim = 32
        self.b_module = Backbone_module(t_dim, x_dim, size)
        self.p_module = PT_Module(self.z_dim)

        self.proj = nn.Linear(self.z_dim, 8)
        
        self.size = size

        self.t_dim = t_dim

    def forward(self, t, x):

        t_tmp, representation = self.b_module.forward(t,x)
        inputs = torch.concat((t_tmp, representation),dim=1)
        out = self.p_module.forward(inputs)
        representation = self.proj(representation)
        # return representation, out
        return  representation, out

    def predict(self, tx):
        self.size = tx.shape[0]
        tx = torch.tensor(tx, dtype=torch.float32)
        t = tx[:, :self.t_dim]
        x = tx[:, self.t_dim:]
        _,out = self.forward(t,x)


        return out.squeeze()

    def get_representation(self, t,x):
        self.size = t.shape[0]
        representation = self.b_module.forward(t,x)
        return representation
 
    def train_model(self, args, bs_opt, pt_opt, train_data, test_data, device):
        self.train()
        train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
        test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
        train_loader = DataLoader(train_data, args.train_bs, shuffle=True)
        # temperature = torch.tensor(args.temperature).to(device)
        alpha = torch.tensor(args.alpha).to(device)
        size = torch.tensor(args.train_bs).to(device)
        # y_epoch = args.y_epoch

        neg_size = torch.tensor(args.neg_size).to(device)
        for y_e in tqdm(range(50)):

            for i, sample in enumerate(train_loader):
                t = sample[:, :args.t_dim]
                x = sample[:, args.t_dim:-1]
                y = sample[:, -1:]

                bs_opt.zero_grad()
                representations, out = self(t,x)

                positive_representations = representations

                loss_positives = Partial_DC(t,x,positive_representations)
                negatives = torch.tensor([0.0]).to(device)
                for neg_i in range(neg_size):
                    negatives += torch.exp(Partial_DC(t,x,representations[torch.randperm(representations.shape[0])]))

                loss_negatives =  torch.log(negatives)

                loss_partial = torch.sqrt(loss_positives**2) - loss_negatives

                loss = loss_partial
                if y_e % 100 == 0:
                    print(f'epoch:{y_e}, loss:{loss.item()} loss_pos:{loss_positives.item()} loss_neg:{loss_negatives.item()}')
                    y_hat = self.predict(test_data[:,:-1]).squeeze()
                    mise = ((test_data[:,-1].squeeze()-y_hat.squeeze())**2).mean()
                    print(f'mise:{mise}')
                loss.backward()
                bs_opt.step()


        for y_e in tqdm(range(50)):

            for i, sample in enumerate(train_loader):
                t = sample[:, :args.t_dim]
                x = sample[:, args.t_dim:-1]
                y = sample[:, -1:]

                pt_opt.zero_grad()
                representations, out = self(t,x)


                loss_y = ((out.squeeze()-y.squeeze())**2).mean()

                loss = loss_y
                if y_e % 10 == 0:
                    print(f'epoch:{y_e}, loss:{loss.item()}')
                    y_hat = self.predict(test_data[:,:-1]).squeeze()
                    mise = ((test_data[:,-1].squeeze()-y_hat.squeeze())**2).mean()
                    print(f'mise:{mise}')
                loss.backward()
                pt_opt.step()


        return self
  
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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
    def __init__(self, t_dim, z_dim) -> None:
        super().__init__()
        self.T_linear1 = nn.Linear(t_dim, 100)
        self.z_dim = z_dim
        self.T_linear2 = nn.Linear(100, self.z_dim)


    def forward(self, t):
        t_out1 = F.leaky_relu(self.T_linear1(t))
        t_out = F.leaky_relu(self.T_linear2(t_out1))

        return t_out

class X_module(nn.Module):
    def __init__(self, x_dim, z_dim) -> None:
        super().__init__()
        self.X_linear1 = nn.Linear(x_dim, 100)
        self.z_dim = z_dim

        self.X_linear2 = nn.Linear(100, self.z_dim)
    
    def forward(self, x):
        x_out = F.leaky_relu(self.X_linear1(x))
        out = F.leaky_relu(self.X_linear2(x_out))

        return out

class Backbone_module(nn.Module):
    def __init__(self, t_dim, x_dim, z_dim, size) -> None:
        super().__init__()
        self.t_dim = t_dim
        self.x_module = X_module(x_dim, z_dim)
        self.t_module = T_module(t_dim, z_dim)
        self.size = size

    def pd_forward(self, t, x):
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        t_tmp = self.t_module.forward(t)
        x_tmp = self.x_module.forward(x)
        
        return t_tmp, x_tmp
    def forward(self, t, x):

        t_tmp, x_tmp = self.pd_forward(t,x)

        return t_tmp, x_tmp



class CRNet(nn.Module):

    def __init__(self, t_dim, x_dim, z_dim, size) -> None:
        super().__init__()
        self.b_module = Backbone_module(t_dim, x_dim, z_dim, size)
        # self.Linear0 = nn.Linear(t_dim+x_dim,128)
        self.z_dim = z_dim
        #ihdp news 8d, data-t-x 16d
        self.proj = nn.Linear(self.z_dim, 8)
        self.Linear1 = nn.Linear(self.z_dim*2, 50)
        # self.Linear1 = nn.Linear(self.z_dim*2, 1)

        # self.Linear1 = nn.Linear(128, 256)

        # self.Linear2 = nn.Linear(50, 50)
        # self.Linear3 = nn.Linear(50, 50)
        self.Linear4 = nn.Linear(50, 1)
        self.size = size

        self.t_dim = t_dim

    def forward(self, t, x):

        t_tmp, representation = self.b_module.forward(t,x)

        # inputs = representation
        # inputs = F.leaky_relu(self.Linear0(torch.cat((t,x),dim=1)))
        # inputs = torch.concat((t,representation),dim=1)
        inputs = torch.concat((t_tmp, representation),dim=1)
        # out = self.Linear1(inputs)
        out = F.leaky_relu(self.Linear1(inputs))
        # out = F.leaky_relu(self.Linear2(out))
        # out = F.leaky_relu(self.Linear3(out))

        out = self.Linear4(out)
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
 
    def train_model(self, args, opt, train_data, test_data, device):
        self.train()

        train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
        test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
        train_loader = DataLoader(train_data, args.train_bs, shuffle=True)

        alpha = torch.tensor(args.alpha).to(device)
        size = torch.tensor(args.train_bs).to(device)

        neg_size = torch.tensor(args.neg_size).to(device)
        # ihdp-1 news-16 50e, data-10-100 200e. data-1-100 100e
        # for y_e in tqdm(range(50)):
        for y_e in tqdm(range(args.n_epochs)):


            for i, sample in enumerate(train_loader):
                t = sample[:, :args.t_dim]
                x = sample[:, args.t_dim:-1]
                y = sample[:, -1:]

                opt.zero_grad()
                # representations = model(t_all,x_all)
                representations, out = self(t,x)

                positive_representations = representations


                loss_positives = Partial_DC(t,x,positive_representations)
                negatives = torch.tensor([0.0]).to(device)
                for neg_i in range(neg_size):
                    negatives += torch.exp(Partial_DC(t,x,representations[torch.randperm(representations.shape[0])]))

                loss_negatives =  torch.log(negatives)

                loss_partial = torch.sqrt(loss_positives**2)
                if neg_size > 0:
                    loss_partial = loss_partial - loss_negatives

                loss_y = ((out.squeeze()-y.squeeze())**2).mean()

                loss = loss_y + alpha * loss_partial
                if y_e % 20 == 0:
                    print(f'epoch:{y_e}, loss:{loss.item()} loss_y:{loss_y.item()} loss_pos:{loss_positives.item()} loss_neg:{loss_negatives.item()}')
                    y_hat = self.predict(test_data[:,:-1]).squeeze()
                    mise = ((test_data[:,-1].squeeze()-y_hat.squeeze())**2).mean()
                    print(f'mise:{mise}')
                loss.backward()
                opt.step()


        return self
  
# TY \perp X | R, include negative samples, weight
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.earlystop import EarlyStopper
import os

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
        self.T_linear1 = nn.Linear(t_dim, z_dim)
        self.z_dim = z_dim
        self.T_linear2 = nn.Linear(z_dim, z_dim)


    def forward(self, t):
        t_out1 = F.elu(self.T_linear1(t))
        t_out = F.elu(self.T_linear2(t_out1))

        return t_out

class X_module(nn.Module):
    def __init__(self, x_dim, z_dim) -> None:
        super().__init__()
        self.X_linear1 = nn.Linear(x_dim, z_dim)
        self.z_dim = z_dim

        self.X_linear2 = nn.Linear(z_dim, z_dim)
    
    def forward(self, x):
        x_out = F.elu(self.X_linear1(x))
        out = F.elu(self.X_linear2(x_out))

        return out

class W_module(nn.Module):
    def __init__(self, z_dim) -> None:
        super().__init__()
        self.W_linear1 = nn.Linear(z_dim, z_dim)
        self.z_dim = z_dim

        self.W_linear2 = nn.Linear(z_dim, 1)
    
    def forward(self, x):
        x_out = F.elu(self.W_linear1(x))
        out = F.softplus(self.W_linear2(x_out))
        out = (out/out.sum())*out.shape[0]
        return out
    
class Backbone_module(nn.Module):
    def __init__(self, t_dim, x_dim, z_dim, size) -> None:
        super().__init__()
        self.t_dim = t_dim
        self.x_module = X_module(x_dim, z_dim)
        self.t_module = T_module(t_dim, z_dim)
        self.proj = nn.Linear(z_dim, 1)
        self.size = size

    def pd_forward(self, t, x):
        if t.ndim == 1:
            t = t.reshape(-1, 1)
        t_tmp = self.t_module.forward(t)
        x_tmp = self.x_module.forward(x)
        
        return t_tmp, x_tmp
    def forward(self, t, x):

        t_tmp, x_tmp = self.pd_forward(t,x)
        representation = self.proj(x_tmp)

        return t_tmp, x_tmp, representation

class FineTuning_module(nn.Module):
    def __init__(self, z_dim) -> None:
        super().__init__()
        self.Linear1 = nn.Linear(z_dim*2, z_dim)
        # self.Linear1 = nn.Linear(z_dim*2, 1)

        # self.Linear1 = nn.Linear(128, 256)

        # self.Linear2 = nn.Linear(50, 50)
        # self.Linear3 = nn.Linear(50, 50)
        self.Linear4 = nn.Linear(z_dim, 1)
    def forward(self, inputs):
        out = F.elu(self.Linear1(inputs))
        # out = self.Linear1(inputs)

        # out = F.elu(self.Linear2(out))
        # out = F.elu(self.Linear3(out))

        out = self.Linear4(out)
        return out

class WDBN(nn.Module):

    def __init__(self, t_dim, x_dim, z_dim, size) -> None:
        super().__init__()
        self.b_module = Backbone_module(t_dim, x_dim, z_dim, size)
        # self.Linear0 = nn.Linear(t_dim+x_dim,128)
        self.z_dim = z_dim
        #ihdp news 8d, data-t-x 16d
        self.proj = nn.Linear(z_dim, 1)
        self.ft = FineTuning_module(z_dim)
        self.w_module = W_module(1)
        self.size = size

        self.t_dim = t_dim

    def forward(self, t, x):

        t_tmp, x_tmp, representation = self.b_module.forward(t,x)

        inputs = torch.concat((t_tmp, x_tmp),dim=1)

        out = self.ft(inputs)

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
    
    def get_w(self,w_init):
        w = self.w_module(w_init)
        return w
 
    def train_model(self, args, b_opt, w_opt, y_opt, train_data, valid_data, test_data, device, save_path):
        self.train()
        early_stopper = EarlyStopper(patience=5, min_delta=0)
        best_val_loss = float('inf')
        train_data = torch.tensor(train_data, dtype=torch.float32).to(device)
        valid_data = torch.tensor(valid_data, dtype=torch.float32).to(device)
        test_data = torch.tensor(test_data, dtype=torch.float32).to(device)
        train_loader = DataLoader(train_data, args.train_bs, shuffle=True)

        alpha = torch.tensor(args.alpha).to(device)
        beta = torch.tensor(args.beta).to(device)
        size = torch.tensor(args.train_bs).to(device)

        neg_size = torch.tensor(args.neg_size).to(device)
        # ihdp-1 news-16 50e, data-10-100 200e. data-1-100 100e
        # for y_e in tqdm(range(50)):
        # 第一个阶段，只训练backbone和proj，loss是loss_partial
        # 第二个阶段，冻结backbone和proj，只训练ft层的w和weighted y
        for b_e in tqdm(range(args.n_epochs)):
            for i, sample in enumerate(train_loader):
                self.train()
                t = sample[:, :args.t_dim]
                x = sample[:, args.t_dim:-1]
                y = sample[:, -1:]
                ty = torch.concat([t,y], dim=1)
                b_opt.zero_grad()
                # w_opt.zero_grad()
                # ft_opt.zero_grad()

                representations, out = self(t,x)

                positive_representations = representations


                loss_positives = Partial_DC(ty,x,positive_representations)
                negatives = torch.tensor([0.0]).to(device)
                for neg_i in range(neg_size):
                    negatives += torch.exp(Partial_DC(ty,x,representations[torch.randperm(representations.shape[0])]))

                loss_negatives = torch.log(negatives)

                loss_partial = loss_positives**2
                if neg_size > 0:
                    loss_partial = loss_partial - loss_negatives
                loss_y = ((out.squeeze()-y.squeeze())**2).mean()

                loss = loss_y + alpha * loss_partial
                loss.backward()
                b_opt.step()
            if b_e % 10 == 0:
                self.eval()
                y_hat = self.predict(test_data[:,:-1]).squeeze()
                valid_loss = ((test_data[:,-1].squeeze()-y_hat.squeeze())**2).mean()
                print(f'epoch:{b_e}, valid_loss:{valid_loss.item()} loss_y:{loss_y.item()} loss_pos:{loss_positives.item()} loss_neg:{loss_negatives.item()}')

        w_init = torch.ones((train_data.shape[0],1)).to(device)
        for w_e in tqdm(range(args.n_epochs)):

            for i, sample in enumerate(train_loader):
                self.train()
                t = sample[:, :args.t_dim]
                x = sample[:, args.t_dim:-1]
                y = sample[:, -1:]
                ty = torch.concat([t,y], dim=1)
                # b_opt.zero_grad()
                w_opt.zero_grad()
                # y_opt.zero_grad()

                representations, out = self(t,x)
                w = self.w_module(representations)
                positive_representations = representations

                # sample_weight = F.softplus(sample_weight)  # Ensure all weights are positive
                # sample_weight = sample_weight / sample_weight.sum()
                # weighted_loss_positives = Partial_DC(w*ty,w*x,w*positive_representations)
                weighted_loss_positives = Partial_DC(ty,x,w*positive_representations)


                weighted_loss_partial = weighted_loss_positives**2

                # weighted_loss_y = (sample_weight.squeeze()*(out.squeeze()-y.squeeze())**2).mean()
                # y_loss = weighted_loss_y
                w_loss = -weighted_loss_partial + beta*(w**2).mean()

                # loss = y_loss - alpha * w_loss
                # loss.backward(retain_graph=True)
                # y_opt.step()
                w_loss.backward()
                w_opt.step()

            if w_e % 20 == 0:
                self.eval()
                y_hat = self.predict(test_data[:,:-1]).squeeze()
                valid_loss = ((test_data[:,-1].squeeze()-y_hat.squeeze())**2).mean()
                print(f'epoch:{w_e}, valid_loss:{valid_loss.item()} loss_pos:{weighted_loss_partial.item()}')
                print(f'w_sum:{w.sum()}, w_mean:{w.mean()}, w_min:{w.min()}, w_max:{w.max()} w**2:{(w**2).mean().item()}, w_loss:{w_loss.item()}')


        for y_e in tqdm(range(args.n_epochs)):

            for i, sample in enumerate(train_loader):
                self.train()
                t = sample[:, :args.t_dim]
                x = sample[:, args.t_dim:-1]
                y = sample[:, -1:]
                ty = torch.concat([t,y], dim=1)
                y_opt.zero_grad()

                representations, out = self(t,x)
                w = self.w_module(representations)
                # positive_representations = representations

                # sample_weight = F.softplus(sample_weight)  # Ensure all weights are positive
                # sample_weight = sample_weight / sample_weight.sum()
                # weighted_loss_positives = Partial_DC(w*ty,w*x,w*positive_representations)

                # weighted_loss_partial = weighted_loss_positives**2

                weighted_loss_y = (w.squeeze()*(out.squeeze()-y.squeeze())**2).mean()
                y_loss = weighted_loss_y
                y_loss.backward()
                y_opt.step()

            if y_e % 20 == 0:
                self.eval()
                y_hat = self.predict(test_data[:,:-1]).squeeze()
                valid_loss = ((test_data[:,-1].squeeze()-y_hat.squeeze())**2).mean()
                print(f'epoch:{w_e}, valid_loss:{valid_loss.item()} y_loss:{weighted_loss_y.item()}')
                print(f'w_sum:{w.sum()}, w_mean:{w.mean()}, w_min:{w.min()}, w_max:{w.max()}')

                # # print(f'mise:{valid_loss}')
                # if valid_loss < best_val_loss:
                #     best_val_loss = valid_loss
                #     # torch.save(self.state_dict(), os.path.join(save_path, 'best_'+str(args.model)+'_model'+str(args.exp)+'.pth'))
                #     print(f'epoch:{y_e}, valid_loss:{valid_loss.item()} loss_y:{loss_y.item()} loss_pos:{loss_positives.item()} loss_neg:{loss_negatives.item()}')
                # self.train()
                # if early_stopper.early_stop(valid_loss):
                #     print(f"epoch: {y_e} --- loss: {loss} --- valid_loss: {valid_loss}")
                #     break


        return self
  
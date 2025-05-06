import os
import torch

import numpy as np
import torch
from sklearn.metrics import pairwise_distances


def regulairize(x):
    return (x - x.min())/(x.max()-x.min())
def setup_seed(seed):
    #  torch.manual_seed(seed)
    #  torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
    #  random.seed(seed)

def log(save_path, txt, file_name='file.txt'):
    with open(os.path.join(save_path, file_name), 'a+') as f:
        f.write(txt)

def rescal(x, xmin, xmax):
    return (x - xmin) / (xmax - xmin)

def test_ihdp(model, args, device, test_data, adrf):
    test_data=test_data
    adrf = adrf
    t_dim = args.t_dim

    
    adrf_hat = np.zeros((adrf.shape[0]))
    for test_id in range(adrf.shape[0]):
        t_tmp = adrf[test_id, :t_dim].reshape((-1,t_dim)).repeat(test_data.shape[0],axis=0)
        x_tmp = test_data[:, args.t_dim:-1]

        t_tmp = torch.tensor(t_tmp, dtype=torch.float32).to(device)
        x_tmp = torch.tensor(x_tmp, dtype=torch.float32).to(device)

        _, _, t_tmp2, x_tmp2, out= model.b_forward(t_tmp,x_tmp)
        y_hat = out.squeeze().mean()
        adrf_hat[test_id] = y_hat
        # print(adrf_hat[test_id], adrf[test_id, -1])
    adrf_mse = ((adrf[:,-1].squeeze()-adrf_hat.squeeze())**2).mean()
    # print(f'adrf_mse: {adrf_mse}')
    # log(args.save_dir, 'adrf_mse:'+str(adrf_mse)+'\n')
    return adrf_mse

def safe_sqrt(x, epsilon=1e-12):
    ''' Numerically safe version of TensorFlow sqrt '''
    return torch.sqrt(x+epsilon)

def pairwise_distance(X, Y=None):
    if Y is None:
        Y = X.copy()
    X_num = np.shape(X)[0]
    Y_num = np.shape(Y)[0]
    # epsilon=1e-12
    # 为什么要加epsilon? 好像是tf的精度问题，先不加

    return np.sum((np.tile(np.expand_dims(X,0),(Y_num, 1, 1)) -np.tile(np.expand_dims(Y,1),(1,X_num, 1)))**2,axis=2)
def pairwise_distance_tensor(X, Y=None):
    if Y is None:
        Y = X[:]
    X_num = X.shape[0]
    Y_num = Y.shape[0]
    # epsilon=1e-12
    # 为什么要加epsilon? 好像是tf的精度问题，先不加
    x_tmp = X.unsqueeze(0).repeat(Y_num, 1, 1)
    y_tmp = Y.unsqueeze(1).repeat(1,X_num, 1)
    return torch.sum((x_tmp - y_tmp)**2,axis=2).sqrt()
def energy_distance(X, Y):
    X_num = np.shape(X)[0]
    Y_num = np.shape(Y)[0]

    exy = np.sum(pairwise_distances(X, Y)) / (float(X_num) * float(Y_num))
    exx = np.sum(pairwise_distances(X, X)) / (float(X_num) * float(X_num))
    eyy = np.sum(pairwise_distances(Y, Y)) / (float(Y_num) * float(Y_num))

    return 2.0 * exy - exx - eyy

    
def distance_covariance(X,Y):
    epsilon=1e-12
    sample_size=np.shape(X)[0]
    
    # pairwise distance of X and pairwise distance of Y
    a_jk = np.sqrt(pairwise_distances(X, X))
    b_jk = np.sqrt(pairwise_distances(Y, Y))
    # a_jk = pairwise_distances(X, Y)
    # b_jk = pairwise_distances(X, Y)

    # 求均值并转化为矩阵
    a_bar_j = np.sum(a_jk, axis=1)/float(sample_size)
    a_bar_j_M = np.repeat(np.reshape(a_bar_j,(sample_size,1)),sample_size,axis=1)

    a_bar_k = np.sum(a_jk, axis=0)/float(sample_size)
    a_bar_k_M = np.repeat(np.reshape(a_bar_k,(1,sample_size)),sample_size,axis=0)
    
    b_bar_j = np.sum(b_jk, axis=1)/float(sample_size)
    b_bar_j_M = np.repeat(np.reshape(b_bar_j,(sample_size,1)),sample_size,axis=1)

    b_bar_k = np.sum(b_jk,axis=0)/float(sample_size)
    b_bar_k_M = np.repeat(np.reshape(b_bar_k,(1,sample_size)),sample_size,axis=0)

    a_bar = np.sum(a_bar_j)/float(sample_size)
    a_bar_M = np.ones((sample_size,sample_size),dtype=np.float32) * a_bar

    b_bar = np.sum(b_bar_j)/float(sample_size)
    b_bar_M = np.ones((sample_size,sample_size),dtype=np.float32) * b_bar

    # dc = np.sum((a_jk-a_bar_j_M-a_bar_k_M+a_bar_M) * (b_jk-b_bar_j_M-b_bar_k_M+b_bar_M))/float(sample_size)**2
    dc = (a_jk-a_bar_j_M-a_bar_k_M+a_bar_M) * (b_jk-b_bar_j_M-b_bar_k_M+b_bar_M)
    return dc


def distance_covariance_test(X,Y):
    sample_size=np.shape(X)[0]
    a_kl=np.sqrt(pairwise_distances(X, X)+1e-12)
    a_bar_k=np.sum(a_kl,axis=1)/sample_size
    a_bar_l=np.sum(a_kl,axis=0)/sample_size
    a_bar=np.sum(a_bar_k)/sample_size
    b_kl=np.sqrt(pairwise_distances(Y, Y)+1e-12)
    b_bar_k=np.sum(b_kl,axis=1)/sample_size
    b_bar_l=np.sum(b_kl,axis=0)/sample_size
    b_bar=np.sum(b_bar_k)/sample_size
    a_bar_k_M=np.repeat(np.reshape(a_bar_k,(sample_size,1)),sample_size,axis=1)
    a_bar_l_M=np.repeat(np.reshape(a_bar_l,(1,sample_size)),sample_size,axis=0)
    a_bar_M=np.ones((sample_size,sample_size))*a_bar
    b_bar_k_M=np.repeat(np.reshape(b_bar_k,(sample_size,1)),sample_size,axis=1)
    b_bar_l_M=np.repeat(np.reshape(b_bar_l,(1,sample_size)),sample_size,axis=0)
    b_bar_M=np.ones((sample_size,sample_size))*b_bar

    dCov = np.sum(np.sum((a_kl-a_bar_k_M-a_bar_l_M+a_bar_M)*(b_kl-b_bar_k_M-b_bar_l_M+b_bar_M)))/sample_size**2
    
    T2 = np.mean(a_kl)*np.mean(b_kl)
    # 
    return sample_size*dCov/T2

def distance_correlation(X,Y):

    d_xx = distance_covariance(X,X).sum()
    d_yy = distance_covariance(Y,Y).sum()
    dcov = distance_covariance(X,Y).sum()
    return dcov/np.sqrt(d_xx*d_yy)
from argparse import ArgumentParser
import os
import random
import warnings
import numpy as np
import torch
from utils import *
from torch.utils.data.dataloader import DataLoader
from torch.optim import SGD, Adam
from models.crnet import CRNet
from models.crnet_only_balance import COB
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
from models.wdbn import WDBN


warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


#  torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = ArgumentParser()

    # Set Hyperparameters
    # i/o
    parser.add_argument('--data_dir', type=str, default='data/version14', help='dir of data')
    parser.add_argument('--save_dir', type=str, default='log/2025-3-21/version14', help='dir to save result')
    parser.add_argument('--n_epochs', type=int, default=100, help='num of epochs to train')

    parser.add_argument('--n_exps', type=int, default=30, help="the number of experiments")
    parser.add_argument("--train_bs", default=2100, type=int, help='train batch size')
    parser.add_argument('--n_samples', type=int, default=3000, help="the number of generated samples")
    parser.add_argument('--n_train', type=int, default=2100, help="the number of samples for training")
    parser.add_argument('--n_val', type=int, default=600, help="the number of samples for training")
    parser.add_argument('--n_test', type=int, default=300, help="the number of samples for training")
    # 1t alpha=1 100x alpha=1 1e-1 2t alpha=2000 1e-2 5t alpha=5000 1e-2 10t alpha=2000，epoch=500 1e-2
    parser.add_argument('--alpha', type=float, default=20, help="hyparameter of double balancing representation")
    parser.add_argument('--beta', type=float, default=1, help="the number of experiments")
    parser.add_argument('--neg_size', type=float, default=1, help="the number of experiments")

    # parser.add_argument('--temperature', type=int, default=1, help="the number of experiments")
    parser.add_argument('--lr', type=float, default=1e-3, help="the number of samples for training")

    # parser.add_argument('--t_dim', type=int, default=1, help="the dimension of treatments")
    parser.add_argument('--z_dim', type=int, default=128, help="the dimension of covariates")

    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--ipm_function', type=str, default='wass', help='use gpu')

    # print train info
    parser.add_argument('--verbose', type=int, default=500, help='print train info freq')
    parser.add_argument('--n_workers', type=int, default=0, help='num of workers')

    args = parser.parse_args()
    # 定义模型
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    # os.environ['CUDA_VISIBLE_DEVICES']='cpu'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    # 训练模型
    # models = ['drnet']
    # models = ['vcnet']
    # models = ['nn','nnv2']
    # models = ['nn']
    # models = ['cob']
    # models = ['crnet']
    # models = ['crnetv3']
    models = ['wdbn']
    # models = ['crnetv5']
    # models = ['cfr-wass']
    # models = ['tarnet']
    # models = ['dragonnet']




    # models = ['nnv2']

    x_dims = [100]
    t_dims = [10]
    # x_dims = [100]
    # t_dims = [2,5,10]
    for t_dim in t_dims:
        args.t_dim = t_dim
        for x_dim in x_dims:
            args.x_dim = x_dim
            data_path = os.path.join(args.data_dir, str(args.t_dim)+'t_'+str(args.x_dim)+'x')

            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            random.seed(0)
            for name in models:
                args.model = name
                save_path = os.path.join(args.save_dir, args.model,
                str(args.train_bs)+'train_bs'+
                str(args.alpha)+'alpha_'+\
                str(args.beta)+'beta'+\
                str(args.n_epochs)+'e_'+\
                str(args.z_dim)+'d_'+\
                str(args.neg_size)+'neg_'+\
                str(args.t_dim)+'t_'\
                +str(args.x_dim)+'x')
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                MISE = np.zeros((args.n_exps,))
                for exp in range(args.n_exps):
                    args.exp = exp
                    print(f'model: {args.model}')

                    train_data = np.load(os.path.join(data_path,str(args.t_dim)+'t_'+
                    str(args.x_dim)+'x_1y_'+str(args.n_samples)+'n_'+str(exp)+'seed.npy'))

                    # valid_data = np.load(os.path.join(data_path,str(args.t_dim)+'t_'+
                    # str(args.x_dim)+'x_1y_'+str(args.n_samples)+'n_'+str(exp)+'seed_valid.npy'))

                    test_data = np.load(os.path.join(data_path,str(args.t_dim)+'t_'+
                    str(args.x_dim)+'x_1y_'+str(args.n_samples)+'n_'+str(exp)+'seed_test.npy'))

                    if args.model == 'crnet':
                        model = CRNet(args.t_dim, args.x_dim, args.z_dim, train_data.shape[0]).to(device)

                        # trick，只训练b_module效果会好得多。
                        opt = Adam(model.b_module.parameters(), lr=args.lr)
                        # opt = Adam(model.parameters(), lr=args.lr)

                        model = model.train_model(args, opt, train_data,test_data, device)

                    if args.model == 'wdbn':
                        model = WDBN(args.t_dim, args.x_dim, args.z_dim, train_data.shape[0]).to(device)
                        # sample_weight = nn.Parameter(torch.ones(train_data.shape[0], 1).to(device), requires_grad=True)
                        # opt = Adam(model.b_module.parameters(), lr=args.lr)
                        b_opt = Adam(model.b_module.parameters(), lr=args.lr)
                        w_opt = Adam(model.w_module.parameters(), lr=args.lr)
                        ft_opt = Adam(model.ft.parameters(), lr=args.lr)

                        model = model.train_model(args, b_opt, w_opt, ft_opt, train_data,test_data, test_data, device, save_path)

                    model.eval()
                    y_hat = model.predict(torch.tensor(test_data[:,:-1],dtype=torch.float32).to(device)).squeeze().cpu().detach().numpy()
                    mise = ((test_data[:,-1]-y_hat)**2).mean()
                    print(mise)
                    MISE[exp] = mise
                    # adrf_mse = ((out.squeeze().cpu().detach().numpy()-adrf[:,-1].squeeze())**2).mean()
                    log(save_path, 'exp: '+str(exp)+'mise: '+str(mise)+ '\n', name+'.txt')

                    # adrf_MSE[exp] = adrf_mse
                    
                
                log(save_path, 'data_'+str(args.t_dim)+'_'+str(args.x_dim)+' ADRF mse: '+\
                    str(MISE.mean().round(3))+' ± '+str(MISE.std().round(2)) +\
                    '\n', name+'.txt')

    # os.system('/root/upload.sh')
            # log(save_path, 'ADRF mse: '+str(adrf_MSE.mean().round(3))+' ± '+str(adrf_MSE.std().round(4)) + '\n', name+'.txt')


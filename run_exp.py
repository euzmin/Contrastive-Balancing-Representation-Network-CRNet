from argparse import ArgumentParser
import os
import random
import warnings
import numpy as np
import torch
from utils import *
from torch.utils.data.dataloader import DataLoader
from models.drnet import Drnet
from models.vcnet import Vcnet
from models.nn import NN
from torch.optim import SGD, Adam
# from models.cr import CRNet
from models.crnet import CRNet
from models.nnv2 import NNv2
from models.crnet_only_balance import COB
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings('ignore')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


#  torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    parser = ArgumentParser()

    # Set Hyperparameters
    # i/o
    parser.add_argument('--data_dir', type=str, default='data/simulation', help='dir of data')
    parser.add_argument('--save_dir', type=str, default='log/2023-8-2/simulation', help='dir to save result')
    parser.add_argument('--n_epochs', type=int, default=50, help='num of epochs to train')

    parser.add_argument('--n_exps', type=int, default=30, help="the number of experiments")
    parser.add_argument("--train_bs", default=1000, type=int, help='train batch size')
    parser.add_argument('--n_samples', type=int, default=3000, help="the number of generated samples")
    parser.add_argument('--n_train', type=int, default=2100, help="the number of samples for training")
    parser.add_argument('--n_val', type=int, default=900, help="the number of samples for training")
    parser.add_argument('--n_test', type=int, default=300, help="the number of samples for training")
    # 1t alpha=1 100x alpha=1 1e-1 2t alpha=2000 1e-2 5t alpha=5000 1e-2 10t alpha=2000，epoch=500 1e-2
    parser.add_argument('--alpha', type=float, default=500, help="the number of experiments")
    parser.add_argument('--neg_size', type=float, default=1, help="the number of experiments")

    # parser.add_argument('--temperature', type=int, default=1, help="the number of experiments")
    parser.add_argument('--lr', type=float, default=1e-2, help="the number of samples for training")

    # parser.add_argument('--t_dim', type=int, default=1, help="the dimension of treatments")
    parser.add_argument('--z_dim', type=int, default=32, help="the dimension of covariates")

    # print train info
    parser.add_argument('--verbose', type=int, default=500, help='print train info freq')
    parser.add_argument('--n_workers', type=int, default=0, help='num of workers')

    args = parser.parse_args()
    # 定义模型
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    # os.environ['CUDA_VISIBLE_DEVICES']='cpu'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 训练模型
    # models = ['drnet']
    # models = ['vcnet']
    # models = ['nn','nnv2']
    # models = ['nn']
    # models = ['cob']
    models = ['crnet']
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
            save_path = os.path.join(args.save_dir, 
            str(args.alpha)+'alpha_'+\
            str(args.n_epochs)+'e_'+\
            str(args.z_dim)+'d_'+\
            str(args.neg_size)+'neg_'+\
            str(args.t_dim)+'t_'\
            +str(args.x_dim)+'x')
            data_path = data_path.replace('/', '\\')
            save_path = save_path.replace('/', '\\')
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            random.seed(0)
            for name in models:
                args.model = name

                MISE = np.zeros((args.n_exps,))
                for exp in range(args.n_exps):

                    print(f'model: {args.model}')

                    train_data = np.load(os.path.join(data_path,str(args.t_dim)+'t_'+
                    str(args.x_dim)+'x_1y_'+str(args.n_samples)+'n_'+str(exp)+'seed.npy'))

                    test_data = np.load(os.path.join(data_path,str(args.t_dim)+'t_'+
                    str(args.x_dim)+'x_1y_'+str(args.n_samples)+'n_'+str(exp)+'seed_test.npy'))

                    if args.model == 'nn':
                        model = NN(args.t_dim+args.x_dim)
                        opt = Adam(model.parameters(), lr=1e-2)
                        model = model.train_model(args, device, opt, train_data, None, None).to(device)
                        
                    if args.model == 'drnet':

                        mm = MinMaxScaler() 
                        # train_data[:,:1] = mm.fit_transform(train_data[:,:1])
                        # test_data[:,:1] = mm.fit_transform(test_data[:,:1])
                        # adrf[:,:-1] = mm.fit_transform(adrf[:,:-1])
                        train_matrix = torch.tensor(train_data, dtype=torch.float32)
                        
                        alpha = 0
                        cfg_density = [(args.x_dim + args.t_dim - 1, 200, 1, 'relu'), (200, 200, 1, 'relu')]
                        num_grid = 10
                        cfg = [(200, 200, 1, 'relu'), (200, 1, 1, 'id')]
                        isenhance = 1

                        model = Drnet(cfg_density, num_grid, cfg, isenhance)
                        model._initialize_weights()
                        momentum = 0.90
                        weight_decay = 0
                        opt = Adam(model.parameters(), lr=1e-2)

                        model = model.train_model(opt, train_matrix)

                    if args.model == 'vcnet':

                        alpha = 0.5
                        cfg_density = [(args.x_dim + args.t_dim - 1, 200, 1, 'relu'), (200, 200, 1, 'relu')]
                        num_grid = 10
                        cfg = [(50, 50, 1, 'relu'), (50, 1, 1, 'id')]
                        degree = 2
                        knots = [0.2, 0.4, 0.6, 0.8]
                        model = Vcnet(cfg_density, num_grid, cfg, degree, knots)
                        momentum = 0.9
                        weight_decay = 1e-7

                        model._initialize_weights()
                        opt = Adam(model.parameters(), lr=1e-2)
                        # opt = SGD(model.parameters(), lr=1e-5, momentum=momentum, weight_decay=weight_decay)
                        mm = MinMaxScaler() 
                        # train_data[:,:1] = mm.fit_transform(train_data[:,:1])
                        # test_data[:,:1] = mm.fit_transform(test_data[:,:1])

                        # adrf[:,:-1] = mm.fit_transform(adrf[:,:-1])
                        train_matrix = torch.tensor(train_data, dtype=torch.float32)
                        
                        model = model.train_model(opt, train_matrix)

                    if args.model == 'crnet':
                        model = CRNet(args.t_dim, args.x_dim, args.z_dim, train_data.shape[0]).to(device)

                        opt = Adam(model.b_module.parameters(), lr=args.lr)

                        model = model.train_model(args, opt, train_data,test_data, device)
                    if args.model == 'cob':
                        model = COB(args.t_dim, args.x_dim, train_data.shape[0]).to(device)
                        bs_opt = Adam(model.b_module.parameters(), lr=args.lr)
                        pt_opt = Adam(model.p_module.parameters(), lr=args.lr)

                        model = model.train_model(args, bs_opt, pt_opt, train_data,test_data, device)

                    torch.save(model.state_dict(), os.path.join(save_path, name+'_'+str(exp)+'.pkl'))
                    args.model = name
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


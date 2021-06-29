import torch
import numpy as np
import torch.nn as nn
import torch.functional as functional
import torch.optim as optim
import matplotlib
import geomloss_arc
import os.path
import sys
import pickle

# sys.path.append('/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/geomloss')

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.distributions.normal import Normal
from Generator import *
from sinkhorn_pointcloud import *
from torch.autograd import grad, Variable, backward
from time import time
import scipy as sp
import scipy.stats
import os
import socket
from datetime import datetime
from Discriminator_ import Discriminator
from torchvision.datasets import mnist
from torch.utils.data import DataLoader, RandomSampler
from torchvision import transforms

from geomloss import SamplesLoss


# from KEOPS_sinkhorn_simple import *

def x_real_builder_8_Gaussian(batch_size):
    sigma = .2
    skel = np.array([
        [2 * np.cos(0 * np.pi / 4), 2 * np.sin(0 * np.pi / 4)],
        [2 * np.cos(1 * np.pi / 4), 2 * np.sin(1 * np.pi / 4)],
        [2 * np.cos(2 * np.pi / 4), 2 * np.sin(2 * np.pi / 4)],
        [2 * np.cos(3 * np.pi / 4), 2 * np.sin(3 * np.pi / 4)],
        [2 * np.cos(4 * np.pi / 4), 2 * np.sin(4 * np.pi / 4)],
        [2 * np.cos(5 * np.pi / 4), 2 * np.sin(5 * np.pi / 4)],
        [2 * np.cos(6 * np.pi / 4), 2 * np.sin(6 * np.pi / 4)],
        [2 * np.cos(7 * np.pi / 4), 2 * np.sin(7 * np.pi / 4)]
    ])
    temp = np.tile(skel, (batch_size // 8 + 1, 1))
    mus = temp[0:batch_size, :]
    m = Normal(torch.FloatTensor([.0]), torch.FloatTensor([sigma]))
    samples = m.sample((batch_size, 2))
    samples = samples.view((batch_size, 2))
    return samples.new(mus) + samples  # * .2


def x_real_builder(batch_size):
    sigma = .02
    skel = np.array([
        [2.0, 2.0],
        [2.0, 1.0],
        [2.0, 0.0],
        [2.0, -1.0],
        [2.0, -2.0],
        [1.0, 2.0],
        [1.0, 1.0],
        [1.0, 0.0],
        [1.0, -1.0],
        [1.0, -2.0],
        [0.0, 2.0],
        [0.0, 1.0],
        [0.0, 0.0],
        [0.0, -1.0],
        [0.0, -2.0],
        [-1.0, 2.0],
        [-1.0, 1.0],
        [-1.0, 0.0],
        [-1.0, -1.0],
        [-1.0, -2.0],
        [-2.0, 2.0],
        [-2.0, 1.0],
        [-2.0, 0.0],
        [-2.0, -1.0],
        [-2.0, -2.0],
    ])
    temp = np.tile(skel, (batch_size // 25 + 1, 1))
    mus = temp[0:batch_size, :]
    m = Normal(torch.FloatTensor([.0]), torch.FloatTensor([sigma]))
    samples = m.sample((batch_size, 2))
    samples = samples.view((batch_size, 2))
    return samples.new(mus) + samples  # * .2


def get_noise_sampler():
    return lambda m, n: torch.randn(m, n).requires_grad_().cuda()


def cosine_distance(x, y):
    input1 = x / x.norm(dim=2)[:, :, None]
    input2 = y / y.norm(dim=2)[:, :, None]
    output = torch.matmul(input1, input2.transpose(1, 2))
    return 1 - output


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='Sinkhorn_GAN_VarPhi_RKHS')
    parser.add_argument("--optim", type=str, default="ema", help="optimization algorithm to use")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--g_lr", type=float, default=2e-4)
    parser.add_argument("--d_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--gamma", type=float, default=0.)
    parser.add_argument("--num_epochs", type=int, default=1000000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--print_interval", type=int, default=1000)
    parser.add_argument("--vis_interval", type=int, default=1000)

    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_hidden_size", type=int, default=384)
    parser.add_argument("--d_num_layers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=666)

    args = parser.parse_args()
    print(torch.__version__)
    ep = 10
    expname = args.exp_name
    current_time = datetime.now().strftime('%Y-%m-%d_%H')
    log_dir = os.path.join('deter_runs', expname + "_" + current_time, 'gamma_' + str(args.gamma), str(ep) + '_fix')
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "args.txt"), "w") as fp:
        for arg in vars(args):
            fp.write("%s:%s \n" % (arg, str(getattr(args, arg))))

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    epsilon = .01

    g_input_size = args.latent_dim

    d_minibatch_size = args.batch_size

    num_epochs = args.num_epochs

    d_learning_rate = args.d_lr
    g_learning_rate = args.g_lr

    noise_data = get_noise_sampler()

    g_hidden_size = args.hidden_size
    g_output_size = 2

    d_input_size = 2
    d_hidden_size = args.d_hidden_size
    if args.gamma == 0.:
        d_output_size = 16
    else:
        d_output_size = 14

    # G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size,
    #               num_layers=args.num_layers).cuda()
    # D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size,
    #                   num_layers=args.d_num_layers).cuda()
    # g_optimizer = torch.optim.SGD(G.parameters(), lr=args.g_lr, weight_decay=0)
    # d_optimizer = torch.optim.SGD(D.parameters(), lr=args.d_lr, weight_decay=0)
    # G.load_state_dict(torch.load(
    #     '/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/runs/Sinkhorn_GAN_VarPhi_01_2020-12-10_23_zichuliu/mix1fac1.0/nonlinear0/gauss_iter_2000_G'))
    # D.load_state_dict(torch.load(
    #     '/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/runs/Sinkhorn_GAN_VarPhi_01_2020-12-10_23_zichuliu/mix1fac1.0/nonlinear0/gauss_iter_2000_D'))
    z_test = torch.rand((d_minibatch_size, g_input_size)).cuda()
    z_test = Variable((z_test - 0.5) * 2)
    images = x_real_builder_8_Gaussian(d_minibatch_size).float().cuda()

    # loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.0001, debias=False, cost=cosine_distance)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.0001, debias=False)
    W1_Loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.0001)

    gammas = ['0.0','0.5','0.9','1.0']
    # iters = ['11']
    iters = ['10000','20000','30000','40000','50000']
    path = '/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/deter_runs/Sinkhorn_GAN_VarPhi_RKHS_2021-05-26_17'

    # plt.subplots(figsize=(40,16))
    # plt.axis('off')
    # for i,gamma in enumerate(gammas):
    #     for j,iter in enumerate(iters):
    #         plt.subplot(4,10,i*10+j+1)
    #         # z_test = torch.load(path + '/gamma_'+gamma+'/10_fix/z_test.pt')
    #         if gamma == '0.0':
    #             path_G = '/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/runs/Sinkhorn_GAN_VarPhi_RKHS_2021-03-12_22'+ '/gamma_'+gamma+'/10_fix/gauss_iter_'+iter+'_G'
    #         elif gamma == '0.5':
    #             path_G = '/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/runs/Sinkhorn_GAN_VarPhi_RKHS_2021-03-15_14'+ '/gamma_'+gamma+'/10_fix/gauss_iter_'+iter+'_G'
    #         else:
    #             path_G = path + '/gamma_'+gamma+'/10_fix/gauss_iter_'+iter+'_G'
    #         G.load_state_dict(torch.load(path_G))
    #         fake_data = G.forward(z_test)
    #         fake_data = [item.data.tolist() for item in fake_data]
    #         plt.axis('off')
    #         for item in fake_data:
    #             plt.plot(item[0], item[1], 'b.')
    #
    # for i,gamma in enumerate(gammas):
    #     for j,iter in enumerate(iters):
    #         plt.subplot(4,10,i*10+j+6)
    #         # z_test = torch.load(path + '/gamma_'+gamma+'/10_fix/z_test.pt')
    #         path_G = path + '/gamma_'+gamma+'/10cosine_fix/gauss_iter_'+iter+'_G'
    #         G.load_state_dict(torch.load(path_G))
    #         fake_data = G.forward(z_test)
    #         fake_data = [item.data.tolist() for item in fake_data]
    #         plt.axis('off')
    #         for item in fake_data:
    #             plt.plot(item[0], item[1], 'b.')
    # plt.savefig(path+'/plot.jpg')


    path = '/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/deter_runs/Sinkhorn_GAN_VarPhi_RKHS_2021-05-26_17'
    inds = ['6','66','666','6666','66666']
    # gammas = ['0.0','0.5','0.9','1.0']
    vars = ['11','11_1']
    stat_dict = dict()
    for i, gamma in enumerate(vars):
        gamma_dict = dict()
        for j, ind in enumerate(inds):
            with open(os.path.join(path,'gamma_0.5',gamma,ind,'SD.pickle'),'rb') as SD:
                SDlist = pickle.load(SD)
            with open(os.path.join(path,'gamma_0.5',gamma,ind,'W1.pickle'),'rb') as W1:
                W1list = pickle.load(W1)
            gamma_dict[ind] = (SDlist,W1list)
        stat_dict[gamma] = gamma_dict.copy()
    SD_stat = dict()
    W1_stat = dict()
    for i, gamma in enumerate(vars):
        gamma_dict = dict()
        dlist = np.asarray([stat_dict[gamma][ind][0][:80000] for ind in inds])
        gamma_dict['mean'] = np.mean(dlist,axis=0)
        gamma_dict['std'] = np.std(dlist,axis=0)
        SD_stat[gamma] = gamma_dict.copy()
        gamma_dict = dict()
        dlist = np.asarray([stat_dict[gamma][ind][1][:80000] for ind in inds])
        gamma_dict['mean'] = np.mean(dlist, axis=0)
        gamma_dict['std'] = np.std(dlist, axis=0)
        W1_stat[gamma] = gamma_dict.copy()
    matplotlib.rcParams.update({'font.size':22})
    temp_t = np.linspace(0,80000-1,80000)
    fig,axes = plt.subplots(2,2,figsize=(80, 20))
    (ax3, ax1), (ax4, ax2) = axes
    for i, gamma in enumerate(vars):
        z = np.log10(SD_stat[gamma]['mean'][:80000])
        dz = 0.434*SD_stat[gamma]['std'][:80000]/SD_stat[gamma]['mean'][:80000]
        ax1.plot(temp_t,z,linewidth=5)
        ax1.fill_between(temp_t,z+dz,z-dz,alpha=0.3)
        ax1.set_xticks([0,20000,40000,60000,80000])
        z = np.log10(W1_stat[gamma]['mean'][:80000])
        dz = 0.434 * W1_stat[gamma]['std'][:80000] / W1_stat[gamma]['mean'][:80000]
        ax2.plot(temp_t, z,linewidth=5)
        ax2.fill_between(temp_t, z+dz,z-dz, alpha=0.3)
        ax2.set_xticks([0,20000,40000,60000,80000])
    matplotlib.rc('font',size=40)
    matplotlib.rc('axes', titlesize=40)
    matplotlib.rc('axes', labelsize=40)
    ax1.set_xlabel('number of generator updates')
    ax1.set_ylabel('log Sinkhorn divergence')
    # ax1.legend(['\u03b3=0.0','\u03b3=0.5','\u03b3=0.9','\u03b3=1.0'])
    ax1.legend(['With Warm-Restart','Without Warm-Restart'])
    ax1.grid(True,color='grey')
    ax1.set_title('Sinkhorn divergence vs generator updates')
    # ax2.legend(['\u03b3=0.0','\u03b3=0.5','\u03b3=0.9','\u03b3=1.0'])
    ax2.legend(['With Warm-Restart','Without Warm-Restart'])
    ax2.set_xlabel('number of generator updates')
    ax2.set_ylabel('log Optimal Transport distance')
    ax2.grid(True,color='grey')
    ax2.set_title('Optimal transport distance vs generator updates')
    fig.savefig(os.path.join('/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/deter_runs/', "stat_new.jpg"))

    plt.close()

    path = '/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/deter_runs/Sinkhorn_GAN_VarPhi_RKHS_2021-05-26_23/gamma_0.5/'
    # inds = ['6','66','666','6666','66666']
    inds = [['66','6666'],['6','666','66666']]
    iters = ['C','D']
    stat_dict = dict()
    for i, iter in enumerate(iters):
        iter_dict = dict()
        for j, ind in enumerate(inds[i]):
            with open(os.path.join(path+'6',ind,'SD.pickle'),'rb') as SD:
                SDlist = pickle.load(SD)
            with open(os.path.join(path+'6', ind, 'W1.pickle'), 'rb') as W1:
                W1list = pickle.load(W1)
            iter_dict[ind] = (SDlist,W1list)
        stat_dict[iter] = iter_dict.copy()
    SD_stat = dict()
    W1_stat = dict()
    stopt = 220000
    for i, iter in enumerate(iters):
        iter_dict = dict()
        dlist = np.asarray([stat_dict[iter][ind][0][:stopt] for ind in inds[i]])
        iter_dict['mean'] = np.mean(dlist,axis=0)
        iter_dict['std'] = np.std(dlist,axis=0)
        SD_stat[iter] = iter_dict.copy()
        iter_dict = dict()
        dlist = np.asarray([stat_dict[iter][ind][1][:stopt] for ind in inds[i]])
        iter_dict['mean'] = np.mean(dlist, axis=0)
        iter_dict['std'] = np.std(dlist, axis=0)
        W1_stat[iter] = iter_dict.copy()
    # matplotlib.rcParams.update({'font.size':22})
    temp_t = np.linspace(0,stopt-1,stopt)
    # fig,(ax1,ax2) = plt.subplots(2,1,figsize=(40, 20))
    for i, iter in enumerate(iters):
        temp_t = np.linspace(0, stopt - 1, stopt)
        z = np.log10(SD_stat[iter]['mean'][:stopt])
        dz = 0.434*SD_stat[iter]['std'][:stopt]/SD_stat[iter]['mean'][:stopt]
        ax3.plot(temp_t,z,linewidth=5)
        ax3.fill_between(temp_t,z+dz,z-dz,alpha=0.3)
        ax3.set_xticks([0,stopt/4,2*stopt/4,3*stopt/4,stopt])
        z = np.log10(W1_stat[iter]['mean'][:stopt])
        dz = 0.434 * W1_stat[iter]['std'][:stopt] / W1_stat[iter]['mean'][:stopt]
        ax4.plot(temp_t, z,linewidth=5)
        ax4.fill_between(temp_t, z+dz,z-dz, alpha=0.3)
        ax4.set_xticks([0,stopt/4,2*stopt/4,3*stopt/4,stopt])
    # z = np.log10(stat_dict['6']['6666'][0][:100000])
    # ax1.plot(temp_t, z)
    # ax1.set_xticks([0, 25000, 50000, 75000, 100000])
    # z = np.log10(stat_dict['6']['6666'][1][:100000])
    # ax2.plot(temp_t, z)
    # ax2.set_xticks([0, 25000, 50000, 75000, 100000])
    #
    ax3.set_xlabel('number of generator updates')
    ax3.set_ylabel('log Sinkhorn divergence')
    ax3.legend(['Convergence', 'Divergence'])
    ax3.grid(True, color='grey')
    ax3.set_title('Sinkhorn divergence vs generator updates')
    ax4.legend(['Convergence', 'Divergence'])
    ax4.set_xlabel('number of generator updates')
    ax4.set_ylabel('log Optimal Transport distance')
    ax4.grid(True, color='grey')
    ax4.set_title('Optimal transport distance vs generator updates')
    fig.savefig(os.path.join('/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/deter_runs/', "Warmrestart_stat_new.jpg"))

    plt.close()







 


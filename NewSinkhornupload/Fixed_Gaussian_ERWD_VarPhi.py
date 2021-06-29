import torch
import numpy as np
import torch.nn as nn
import torch.functional as functional
import torch.optim as optim
import matplotlib

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


def x_real_builder(batch_size):
    sigma = .1
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


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='ERWD_VarPhi_3')
    parser.add_argument("--optim", type=str, default="ema", help="optimization algorithm to use")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--g_lr", type=float, default=1e-3)
    parser.add_argument("--d_lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--grad_penalty", type=float, default=0)
    parser.add_argument("--mix_metric_flag", type=int, default=1)
    parser.add_argument("--nonlinear_OT_flag", type=int, default=0)
    parser.add_argument("--nonlinear_fac", type=float, default=1.0)

    parser.add_argument("--num_epochs", type=int, default=50001)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--print_interval", type=int, default=1000)
    parser.add_argument("--vis_interval", type=int, default=1000)

    parser.add_argument("--latent_dim", type=int, default=32)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_hidden_size", type=int, default=128)
    parser.add_argument("--d_num_layers", type=int, default=6)
    parser.add_argument("--npseed", type=int, default=0)
    parser.add_argument("--pyseed", type=int, default=100)

    args = parser.parse_args()

    expname = args.exp_name
    current_time = datetime.now().strftime('%Y-%m-%d_%H')
    log_dir = os.path.join('runs', expname + "_" + current_time + "_" + socket.gethostname(),
                           'mix' + str(args.mix_metric_flag) + 'fac' + str(args.nonlinear_fac),
                           'nonlinear' + 'epsilon001_gamma_05')
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "args.txt"), "w") as fp:
        for arg in vars(args):
            fp.write("%s:%s \n" % (arg, str(getattr(args, arg))))

    np.random.seed(args.npseed)
    torch.random.manual_seed(args.npseed)

    epsilon = .01

    d_minibatch_size = args.batch_size

    num_epochs = args.num_epochs

    d_learning_rate = args.d_lr

    d_input_size = 2
    d_hidden_size = args.d_hidden_size
    d_output_size = 4

    D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size,
                      num_layers=args.d_num_layers).cuda()
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.d_lr, weight_decay=args.weight_decay)

    images = x_real_builder(d_minibatch_size).float().cuda()

    g_size = list(images.shape)
    g = torch.randn(g_size, requires_grad=True).cuda()

    print(images)
    W1LOSS = list()
    for epoch in range(num_epochs):
        if args.mix_metric_flag:
            nfac = args.nonlinear_fac
            D_fake = D.forward(g)
            D_real = D.forward(images)
            D_fake = D_fake.view(D_fake.shape[0], -1)
            D_real = D_real.view(D_real.shape[0], -1)

            if epoch % 10 == 0:
                # for param in D.parameters():
                #     param.requires_grad = True

                loss, _ = ERWD_normalized(g, images, D_fake, D_real, epsilon, d_minibatch_size,
                                                    500, nfac=nfac)
                d_optimizer.zero_grad()
                grad_penalty = 0
                if args.grad_penalty:
                    grad_penalty = D.get_penalty(images, g)
                    D_loss = -loss + args.grad_penalty * grad_penalty
                    print('D_Loss:%f,Grad Penalty:%f' % (-D_loss.data.tolist(), grad_penalty.data.tolist()))
                else:
                    D_loss = -loss
                    print('D_Loss:%f' % (-D_loss.data.tolist()))
                # print('D_Loss:%f,Grad Penalty:%f' % (-D_loss.data.tolist(), grad_penalty.data.tolist()))
                D_loss.backward()
                d_optimizer.step()

            else:
                # for param in D.parameters():
                #     param.requires_grad = False
                g_loss, _ = ERWD_normalized(g, images, D_fake, D_real, epsilon, d_minibatch_size,
                                                      500, nfac=nfac)

                gradients = torch.autograd.grad(g_loss,g)
                g = g - gradients[0]
                # print(g_loss.data.tolist())+
    #
    #     elif args.nonlinear_OT_flag:
    #
    #         D_fake = D.forward(generated_imgs)
    #         D_real = D.forward(images)
    #         D_fake = D_fake.view(D_fake.shape[0], -1)
    #         D_real = D_real.view(D_real.shape[0], -1)
    #
    #         if epoch % 5 == 0:
    #             for param in D.parameters():
    #                 param.requires_grad = True
    #             for param in G.parameters():
    #                 param.requires_grad = False
    #
    #             loss, _ = sinkhorn_normalized(D_fake, D_real, epsilon, d_minibatch_size, 500)
    #             d_optimizer.zero_grad()
    #             grad_penalty = 0
    #             if args.grad_penalty:
    #                 grad_penalty = D.get_penalty(images, generated_imgs)
    #                 D_loss = -loss + args.grad_penalty * grad_penalty
    #             else:
    #                 D_loss = -loss
    #             D_loss.backward()
    #             d_optimizer.step()
    #             print('D_Loss:%f,Grad Penalty:%f' % (-D_loss.data.tolist(), grad_penalty.data.tolist()))
    #
    #         else:
    #             for param in D.parameters():
    #                 param.requires_grad = False
    #             for param in G.parameters():
    #                 param.requires_grad = True
    #             g_loss, _ = sinkhorn_normalized(D_fake, D_real, epsilon, d_minibatch_size, 500)
    #             g_optimizer.zero_grad()
    #             g_loss.backward()
    #             g_optimizer.step()
    #             # print(g_loss.data.tolist())
    #     else:
    #         g_loss, _ = sinkhorn_normalized(generated_imgs, images, epsilon, d_minibatch_size, 500)
    #         g_optimizer.zero_grad()
    #         g_loss.backward()
    #         g_optimizer.step()
    #         # print(g_loss.data.tolist())
    #
        if epoch % 100 == 0:
            fake_data = g
            fake_data = [item.data.tolist() for item in fake_data]
            real_data = [item.data.tolist() for item in images]
            X = [-2, -1, 0., 1, 2]
            Y = [-2, -1, 0., 1, 2]

            w1loss = Wasserstein1(g, images, 0.001, d_minibatch_size, 500)
            W1LOSS.append(w1loss.data.tolist())

            fig, axes = plt.subplots(1, 1,figsize=(20, 20))
            for x in X:
                for y in Y:
                    axes.plot(x, y, 'go')
            for item in fake_data:
                axes.plot(item[0], item[1], 'b.')
            for item in real_data:
                axes.plot(item[0], item[1], 'rx',alpha=0.5)

            axes.grid()
            fig.savefig(os.path.join(log_dir, "gauss_iter_{0}_W1_{1:.4f}.jpg".format(epoch,w1loss)))

            plt.close()
#
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
    sigma = .01
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
    parser.add_argument("--exp_name", type=str, default='Sinkhorn_GAN_VarPhi_001')
    parser.add_argument("--optim", type=str, default="ema", help="optimization algorithm to use")
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--g_lr", type=float, default=1e-3)
    parser.add_argument("--d_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=2e-5)
    parser.add_argument("--grad_penalty", type=float, default=0)
    parser.add_argument("--mix_metric_flag", type=int, default=1)
    parser.add_argument("--nonlinear_OT_flag", type=int, default=0)
    parser.add_argument("--nonlinear_fac", type=float, default=1.0)

    parser.add_argument("--num_epochs", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--print_interval", type=int, default=1000)
    parser.add_argument("--vis_interval", type=int, default=1000)

    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--d_hidden_size", type=int, default=384)
    parser.add_argument("--d_num_layers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    expname = args.exp_name
    current_time = datetime.now().strftime('%Y-%m-%d_%H')
    log_dir = os.path.join('runs', expname + "_" + current_time + "_" + socket.gethostname(),
                           'mix' + str(args.mix_metric_flag) + 'fac' + str(args.nonlinear_fac),
                           'nonlinear' + str(args.nonlinear_OT_flag))
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
    d_output_size = 32

    G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size,
                  num_layers=args.num_layers).cuda()
    D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size,
                      num_layers=args.d_num_layers).cuda()
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.g_lr, weight_decay=args.weight_decay)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.d_lr, weight_decay=args.weight_decay)

    z_test = torch.rand((d_minibatch_size, g_input_size)).cuda()
    z_test = Variable((z_test - 0.5) * 2)

    GRADS = list()
    SLOSS = list()
    W1LOSS = list()
    images_test = x_real_builder(d_minibatch_size)
    for epoch in range(num_epochs):
        images = list()
        generated_imgs = list()
        for i in range(2):
            z = torch.rand((d_minibatch_size, g_input_size)).cuda()
            z = Variable((z - 0.5) * 2)
            images.append(x_real_builder(d_minibatch_size).float().cuda())
            generated_imgs.append(G.forward(z))

        if args.mix_metric_flag:
            nfac = args.nonlinear_fac
            D_fake = list()
            D_real = list()
            for i in range(2):
                D_fake_temp = D.forward(generated_imgs[i])
                D_real_temp = D.forward(images[i])
                D_fake.append(D_fake_temp.view(D_fake_temp.shape[0], -1))
                D_real.append(D_real_temp.view(D_real_temp.shape[0], -1))

            if epoch % 10 == 0:
                for param in D.parameters():
                    param.requires_grad = True
                for param in G.parameters():
                    param.requires_grad = False

                loss, _ = mixed_sinkhorn_normalized_unbiased_1(generated_imgs, images, D_fake, D_real, epsilon, d_minibatch_size,
                                                    500, nfac=nfac)
                d_optimizer.zero_grad()
                grad_penalty = 0
                if args.grad_penalty:
                    grad_penalty = D.get_penalty(images, generated_imgs)
                    D_loss = -loss + args.grad_penalty * grad_penalty
                    print('D_Loss:%f,Grad Penalty:%f' % (-D_loss.data.tolist(), grad_penalty.data.tolist()))
                else:
                    D_loss = -loss
                    print('D_Loss:%f' % (-D_loss.data.tolist()))
                D_loss.backward(retain_graph=True)
                d_optimizer.step()

            for param in D.parameters():
                param.requires_grad = False
            for param in G.parameters():
                param.requires_grad = True
            g_loss, _ = mixed_sinkhorn_normalized_unbiased_1(generated_imgs, images, D_fake, D_real, epsilon, d_minibatch_size,
                                                  500, nfac=nfac)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # temp_grad = list()
            # for p in G.parameters():
            #     temp_grad.append(p.grad.norm().data.tolist())
            # avg_grad = sum(temp_grad) / len(temp_grad)
            GRADS.append(1e-3)
            SLOSS.append(g_loss.data.tolist())

            # w1loss = Wasserstein1(generated_imgs[0], images[0], 0.01, d_minibatch_size, 500)
            # W1LOSS.append(w1loss.data.tolist())

        elif args.nonlinear_OT_flag:

            D_fake = list()
            D_real = list()
            for i in range(4):
                D_fake_temp = D.forward(generated_imgs[i])
                D_real_temp = D.forward(images[i])
                D_fake.append(D_fake_temp.view(D_fake_temp.shape[0], -1))
                D_real.append(D_real_temp.view(D_real_temp.shape[0], -1))

            if epoch % 5 == 0:
                for param in D.parameters():
                    param.requires_grad = True
                for param in G.parameters():
                    param.requires_grad = False

                loss, _ = sinkhorn_normalized(D_fake, D_real, epsilon, d_minibatch_size, 500)
                d_optimizer.zero_grad()
                grad_penalty = 0
                if args.grad_penalty:
                    grad_penalty = D.get_penalty(images, generated_imgs)
                    D_loss = -loss + args.grad_penalty * grad_penalty
                else:
                    D_loss = -loss
                D_loss.backward()
                d_optimizer.step()
                print('D_Loss:%f,Grad Penalty:%f' % (-D_loss.data.tolist(), grad_penalty.data.tolist()))

            else:
                for param in D.parameters():
                    param.requires_grad = False
                for param in G.parameters():
                    param.requires_grad = True
                g_loss, _ = sinkhorn_normalized(D_fake, D_real, epsilon, d_minibatch_size, 500)
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                # print(g_loss.data.tolist())
        else:
            g_loss, _ = sinkhorn_normalized(generated_imgs, images, epsilon, d_minibatch_size, 500)
            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()
            # print(g_loss.data.tolist())

        if epoch % 5 == 0:
            images = images_test
            generated_imgs = G.forward(z_test)
            fake_data = [item.data.tolist() for item in generated_imgs]
            real_data = [item.data.tolist() for item in images]
            X = [-2, -1, 0., 1, 2]
            Y = [-2, -1, 0., 1, 2]
            w1loss = Wasserstein1(generated_imgs, images, 0.01, d_minibatch_size, 500)
            W1LOSS.append(w1loss.data.tolist())
            # fig, axes = plt.subplots(1, 1)
            # for x in X:
            #     for y in Y:
            #         axes.plot(x, y, 'go')
            # for item in fake_data:
            #     axes.plot(item[0], item[1], 'b.')
            # for item in real_data:
            #     axes.plot(item[0], item[1], 'r.')
            #
            # axes.grid()
            # fig.savefig(os.path.join(log_dir, "gauss_iter_%i.jpg" % epoch))
            #
            # plt.close()

            fig, axes = plt.subplots(2, 2, figsize=(15, 15))
            (ax1, ax2), (ax3, ax4) = axes
            temp_t = np.linspace(0, epoch + 1, epoch + 1)
            temp_t_2 = np.linspace(0, len(W1LOSS), len(W1LOSS))

            for x in X:
                for y in Y:
                    ax1.plot(x, y, 'go')
            for item in fake_data:
                ax1.plot(item[0], item[1], 'b.')
            for item in real_data:
                ax1.plot(item[0], item[1], 'r.')

            ax1.grid()
            ax1.set_title('Point Clouds')

            ax2.semilogy(temp_t, GRADS)
            ax2.set_title('Gradient')
            ax3.semilogy(temp_t, SLOSS)
            ax3.set_title('Sinkhorn divergence')
            ax4.semilogy(temp_t_2, W1LOSS)
            ax4.set_title('1-Wasserstein distance')
            ax4.annotate('%f' % W1LOSS[-1],
                         xy=(.88, .15), xycoords='figure fraction',
                         horizontalalignment='left', verticalalignment='top',
                         fontsize=15)
            fig.savefig(os.path.join(log_dir, "gauss_iter_%i.jpg" % epoch))

            plt.close()

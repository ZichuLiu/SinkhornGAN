import torch
import numpy as np
import torch.nn as nn
import torch.functional as functional
import torch.optim as optim
import matplotlib
import geomloss_arc
import os.path
import sys

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
    sigma = .02
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


def cosine_similarity(x, y):
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
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--grad_penalty", type=float, default=0)
    parser.add_argument("--mix_metric_flag", type=int, default=1)
    parser.add_argument("--nonlinear_OT_flag", type=int, default=0)
    parser.add_argument("--nonlinear_fac", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.)
    parser.add_argument("--num_epochs", type=int, default=1000000)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--print_interval", type=int, default=1000)
    parser.add_argument("--vis_interval", type=int, default=1000)

    parser.add_argument("--latent_dim", type=int, default=16)
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
    log_dir = os.path.join('runs', expname + "_" + current_time + "_" + socket.gethostname(),
                           'mix' + str(args.mix_metric_flag) + 'fac' + str(args.nonlinear_fac),
                           'nonlinear' + 'gamma_' + str(args.gamma), str(ep)+'test')
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
        d_output_size = 8
    else:
        d_output_size = 6

    G = Generator(input_size=g_input_size, hidden_size=g_hidden_size, output_size=g_output_size,
                  num_layers=args.num_layers).cuda()
    D = Discriminator(input_size=d_input_size, hidden_size=d_hidden_size, output_size=d_output_size,
                      num_layers=args.d_num_layers).cuda()
    g_optimizer = torch.optim.Adam(G.parameters(), lr=args.g_lr, weight_decay=args.weight_decay)
    d_optimizer = torch.optim.Adam(D.parameters(), lr=args.d_lr, weight_decay=args.weight_decay)
    # G.load_state_dict(torch.load(
    #     '/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/runs/Sinkhorn_GAN_VarPhi_01_2020-12-10_23_zichuliu/mix1fac1.0/nonlinear0/gauss_iter_2000_G'))
    # D.load_state_dict(torch.load(
    #     '/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/runs/Sinkhorn_GAN_VarPhi_01_2020-12-10_23_zichuliu/mix1fac1.0/nonlinear0/gauss_iter_2000_D'))
    z_test = torch.rand((1000, g_input_size)).cuda()
    z_test = Variable((z_test - 0.5) * 2)
    images = x_real_builder_8_Gaussian(1000).float().cuda()

    # loss = SamplesLoss(loss="sinkhorn", p=2, blur=np.sqrt(epsilon), debias=False,cost=cosine_similarity)
    loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.01, debias=False)
    W1_Loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.01)

    GRADS_G = list()
    GRADS_D = list()
    SLOSS = list()
    W1LOSS = list()
    PARAM_G = list()
    PARAM_D = list()
    DELTA_G = list()
    DELTA_D = list()

    temp_p = list()
    for p in D.parameters():
        temp_p.append(p.data.view(-1))
    params = torch.cat(temp_p)
    TEMP_PARAM_D = torch.zeros_like(params)

    temp_p = list()
    for p in G.parameters():
        temp_p.append(p.data.view(-1))
    params = torch.cat(temp_p)
    TEMP_PARAM_G = torch.zeros_like(params)

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        # if epoch == 20000:
        #     print('Switch optimizer')
        #     g_optimizer = torch.optim.SGD(G.parameters(), lr=args.g_lr * 100, weight_decay=0)
        #     d_optimizer = torch.optim.SGD(D.parameters(), lr=args.d_lr * 100, weight_decay=0)
        z_1 = torch.rand((d_minibatch_size, g_input_size)).cuda()
        z_1 = Variable((z_1 - 0.5) * 2)
        z_2 = torch.rand((d_minibatch_size, g_input_size)).cuda()
        z_2 = Variable((z_2 - 0.5) * 2)
        images_1 = x_real_builder_8_Gaussian(d_minibatch_size).float().cuda()
        images_2 = x_real_builder_8_Gaussian(d_minibatch_size).float().cuda()
        # z = z_test
        generated_imgs_1 = G.forward(z_1)
        generated_imgs_2 = G.forward(z_2)

        if args.mix_metric_flag:
            nfac = args.nonlinear_fac
            gamma = args.gamma
            D_fake_1 = D.forward(generated_imgs_1)
            D_real_1 = D.forward(images_1)
            D_fake_1 = D_fake_1.view(D_fake_1.shape[0], -1)
            D_real_1 = D_real_1.view(D_real_1.shape[0], -1)

            D_fake_2 = D.forward(generated_imgs_2)
            D_real_2 = D.forward(images_2)
            D_fake_2 = D_fake_2.view(D_fake_2.shape[0], -1)
            D_real_2 = D_real_2.view(D_real_2.shape[0], -1)
            if gamma != 0 and gamma != 1:
                concat_x_1 = torch.cat((np.sqrt(gamma) * generated_imgs_1, np.sqrt(1 - gamma) * D_fake_1), 1)
                concat_y_1 = torch.cat((np.sqrt(gamma) * images_1, np.sqrt(1 - gamma) * D_real_1), 1)
                concat_x_2 = torch.cat((np.sqrt(gamma) * generated_imgs_2, np.sqrt(1 - gamma) * D_fake_2), 1)
                concat_y_2 = torch.cat((np.sqrt(gamma) * images_2, np.sqrt(1 - gamma) * D_real_2), 1)
            elif gamma == 0:
                concat_x_1 = D_fake_1
                concat_y_1 = D_real_1
                concat_x_2 = D_fake_2
                concat_y_2 = D_real_2
            else:
                concat_x_1 = generated_imgs_1
                concat_y_1 = images_1
                concat_x_2 = generated_imgs_2
                concat_y_2 = images_2

            if epoch % ep == 0:
                if gamma == 1.:
                    GRADS_D.append(1e-4)
                    PARAM_D.append(0)
                    DELTA_D.append(0)
                else:
                    for param in G.parameters():
                        param.requires_grad = False
                    # WX1Y1, _ = ERWD_normalized(generated_imgs_1, images_1, D_fake_1, D_real_1, epsilon,
                    #                            d_minibatch_size,
                    #                            100, nfac=nfac)
                    # WX2Y2, _ = ERWD_normalized(generated_imgs_2, images_2, D_fake_2, D_real_2, epsilon,
                    #                            d_minibatch_size,
                    #                            100, nfac=nfac)
                    # WX1Y2, _ = ERWD_normalized(generated_imgs_2, images_1, D_fake_2, D_real_1, epsilon,
                    #                            d_minibatch_size,
                    #                            100, nfac=nfac)
                    # WX2Y1, _ = ERWD_normalized(generated_imgs_1, images_2, D_fake_1, D_real_2, epsilon,
                    #                            d_minibatch_size,
                    #                            100, nfac=nfac)
                    # WYY, _ = ERWD_normalized(generated_imgs_1, generated_imgs_2, D_fake_1, D_fake_2, epsilon,
                    #                          d_minibatch_size,
                    #                          100, nfac=nfac)
                    # WXX, _ = ERWD_normalized(images_1, images_2, D_real_1, D_real_2, epsilon, d_minibatch_size,
                    #                          100, nfac=nfac)
                    #
                    # neg_loss = -(WX1Y1 + WX1Y2 + WX2Y1 + WX2Y2 - 2 * WYY - 2 * WXX)

                    # D_loss = neg_loss
                    # D_loss = -loss
                    # print('D_Loss:%f' % (neg_loss.data.tolist()))

                    # # loss = SamplesLoss(loss='sinkhorn', p=2, blur=.5)
                    d_optimizer.zero_grad()
                    #     grad_penalty = 0
                    if args.grad_penalty:
                        grad_penalty = D.get_penalty(images, generated_imgs_1)
                        D_loss = -loss + args.grad_penalty * grad_penalty
                        print('D_Loss:%f,Grad Penalty:%f' % (-D_loss.data.tolist(), grad_penalty.data.tolist()))
                    else:
                        # loss = ERWD_normalized()
                        # loss = SamplesLoss(loss='sinkhorn', p=2, blur=epsilon, debias=False)
                        # loss = KEOPS_sinkhorn_divergence(concat_x, concat_y)
                        # D_loss = -mixed_sinkhorn_normalized()
                        neg_loss = -loss(concat_x_1, concat_y_1)
                        neg_loss = -loss(concat_x_1, concat_y_2) + neg_loss
                        neg_loss = -loss(concat_x_2, concat_y_1) + neg_loss
                        neg_loss = -loss(concat_x_2, concat_y_2) + neg_loss
                        neg_loss = loss(concat_x_1, concat_x_2) * 2 + neg_loss
                        neg_loss = loss(concat_y_1, concat_y_2) * 2 + neg_loss

                        D_loss = neg_loss
                        # D_loss = -loss
                        print('D_Loss:%f' % (-D_loss.data.tolist()))
                    D_loss.backward()
                    d_optimizer.step()
                    temp_grad = list()
                    temp_param = list()
                    for p in D.parameters():
                        temp_grad.append(p.grad.view(-1))
                        temp_param.append(p.data.view(-1))
                    grads = torch.cat(temp_grad)
                    param = torch.cat(temp_param)
                    param_delta = param - TEMP_PARAM_D
                    param_delta_norm = param_delta.norm().data.tolist()
                    TEMP_PARAM_D = param.detach()
                    grads_norm = grads.norm().data.tolist()
                    param_norm = param.norm().data.tolist()
                    if grads_norm == 0:
                        GRADS_D.append(1e-4)
                        PARAM_D.append(param_norm)
                        DELTA_D.append(param_delta_norm)
                    else:
                        GRADS_D.append(grads_norm)
                        PARAM_D.append(param_norm)
                        DELTA_D.append(param_delta_norm)
                    print(grads_norm)

                    for param in G.parameters():
                        param.requires_grad = True

                    del neg_loss
                    torch.cuda.empty_cache()

            # for param in D.parameters():
            #     param.requires_grad = False
            # for param in G.parameters():
            #     param.requires_grad = True
            # g_loss, _ = mixed_sinkhorn_normalized(generated_imgs, images, D_fake, D_real, epsilon, d_minibatch_size,
            #                                       200, nfac=nfac)
            else:
                for param in D.parameters():
                    param.requires_grad = False
                # WX1Y1, _ = ERWD_normalized(generated_imgs_1, images_1, D_fake_1, D_real_1, epsilon, d_minibatch_size,
                #                            100, nfac=nfac)
                # WX2Y2, _ = ERWD_normalized(generated_imgs_2, images_2, D_fake_2, D_real_2, epsilon, d_minibatch_size,
                #                            100, nfac=nfac)
                # WX1Y2, _ = ERWD_normalized(generated_imgs_2, images_1, D_fake_2, D_real_1, epsilon, d_minibatch_size,
                #                            100, nfac=nfac)
                # WX2Y1, _ = ERWD_normalized(generated_imgs_1, images_2, D_fake_1, D_real_2, epsilon, d_minibatch_size,
                #                            100, nfac=nfac)
                # WYY, _ = ERWD_normalized(generated_imgs_1, generated_imgs_2, D_fake_1, D_fake_2, epsilon,
                #                          d_minibatch_size,
                #                          100, nfac=nfac)
                # WXX, _ = ERWD_normalized(images_1, images_2, D_real_1, D_real_2, epsilon, d_minibatch_size,
                #                          100, nfac=nfac)
                #
                # pos_loss = WX1Y1 + WX1Y2 + WX2Y1 + WX2Y2 - 2 * WYY - 2 * WXX
                # loss = SamplesLoss(loss='sinkhorn', p=2, blur=epsilon, debias=False)
                # g_loss = KEOPS_sinkhorn_divergence(concat_x, concat_y)
                pos_loss = loss(concat_x_1, concat_y_1)
                pos_loss = loss(concat_x_1, concat_y_2) + pos_loss
                pos_loss = loss(concat_x_2, concat_y_1) + pos_loss
                pos_loss = loss(concat_x_2, concat_y_2) + pos_loss
                pos_loss = -loss(concat_x_1, concat_x_2) * 2 + pos_loss
                pos_loss = -loss(concat_y_1, concat_y_2) * 2 + pos_loss

                g_loss = pos_loss
                # g_loss = loss
                g_optimizer.zero_grad()
                g_loss.backward()
                temp_grad = list()
                temp_param = list()
                for p in G.parameters():
                    temp_grad.append(p.grad.view(-1))
                    temp_param.append(p.data.view(-1))
                grads = torch.cat(temp_grad)
                param = torch.cat(temp_param)
                param_delta = param - TEMP_PARAM_G
                param_delta_norm = param_delta.norm().data.tolist()
                TEMP_PARAM_G = param.detach()
                grads_norm = grads.norm().data.tolist()
                param_norm = param.norm().data.tolist()
                print(grads_norm)
                GRADS_G.append(grads_norm)
                PARAM_G.append(param_norm)
                DELTA_G.append(param_delta_norm)

                g_optimizer.step()

                SLOSS.append(g_loss.data.tolist())

                w1loss = W1_Loss(generated_imgs_1, images_1)
                W1LOSS.append(w1loss.data.tolist())
                for param in D.parameters():
                    param.requires_grad = True

        if epoch % 100 == 0 and epoch != 0:
            fake_data = G.forward(z_test)
            fake_data = [item.data.tolist() for item in fake_data]
            real_data = [item.data.tolist() for item in images]
            # X = [-2, -1, 0., 1, 2]
            # Y = [-2, -1, 0., 1, 2]
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

            fig, axes = plt.subplots(2, 5, figsize=(50, 20))
            (ax1, ax2, ax3, ax4, ax5), (ax6, ax7, ax8, ax9, ax10) = axes
            temp_t = np.linspace(0, len(SLOSS) - 1, len(SLOSS))
            temp_G = np.linspace(0, len(GRADS_G) - 1, len(GRADS_G))
            temp_D = np.linspace(0, len(GRADS_D) - 1, len(GRADS_D))
            temp_d_D = np.linspace(0, len(GRADS_D) - 2, len(GRADS_D) - 1)
            temp_d_G = np.linspace(0, len(GRADS_G) - 2, len(GRADS_G) - 1)

            # for x in X:
            #     for y in Y:
            #         ax1.plot(x, y, 'go')
            for item in fake_data:
                ax1.plot(item[0], item[1], 'b.')
            # for item in real_data:
            #     ax1.plot(item[0], item[1], 'rx',alpha=0.5)

            ax1.grid()
            ax1.set_title('Point Clouds')

            ax2.plot(temp_t, SLOSS)
            ax2.set_title('Sinkhorn divergence')
            ax3.semilogy(temp_t, W1LOSS)
            ax3.set_title('1-Wasserstein distance')
            ax3.annotate('%f' % W1LOSS[-1],
                         xy=(.88, .55), xycoords='figure fraction',
                         horizontalalignment='left', verticalalignment='top',
                         fontsize=15)
            ax4.set_title('Parameter Norm of G')
            ax4.semilogy(temp_G, PARAM_G)
            ax5.set_title('Parameter Update of G')
            ax5.semilogy(temp_d_G, DELTA_G[1:])
            ax6.semilogy(temp_G, GRADS_G)
            ax6.set_title('Gradient Norm of G')
            ax7.semilogy(temp_D, GRADS_D)
            ax7.set_title('Gradient Norm of D')
            temp2_D = temp_D * (ep - 1) - 1
            temp2_D[0] = 0
            ax8.semilogy(temp2_D, GRADS_D, color='orange', label='Grad Norm D')
            ax8.semilogy(temp_G, GRADS_G, color='blue', label='Grad Norm G')
            ax8.legend(loc='upper right')
            ax8.set_title('Gradient')
            ax9.set_title('Parameter Norm of D')
            ax9.semilogy(temp_D, PARAM_D)
            ax10.set_title('Parameter Update of D')
            ax10.semilogy(temp_d_D, DELTA_D[1:])
            fig.savefig(os.path.join(log_dir, "gauss_iter_%i.jpg" % epoch))

            plt.close()

        if epoch % 2000 == 0 and epoch != 0:
            torch.save(G.state_dict(), os.path.join(log_dir, "gauss_iter_%i_G" % epoch))
            torch.save(D.state_dict(), os.path.join(log_dir, "gauss_iter_%i_D" % epoch))

import argparse
import logging
import time
import os
import torch
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image

# from networks import Generator, Discriminator
from DCGAN import Generator, Discriminator
from utils import get_data_loader, generate_images
from geomloss import SamplesLoss

def cosine_distance(x, y):
    input1 = x / x.norm(dim=2)[:, :, None]
    input2 = y / y.norm(dim=2)[:, :, None]
    output = torch.matmul(input1, input2.transpose(1, 2))
    return 1 - output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DCGANS MNIST')
    parser.add_argument('--num-epochs', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--ndf', type=int, default=64, help='Number of features to be used in Discriminator network')
    parser.add_argument('--ngf', type=int, default=64, help='Number of features to be used in Generator network')
    parser.add_argument('--nz', type=int, default=128, help='Size of the noise')
    parser.add_argument('--d-lr', type=float, default=0.0002, help='Learning rate for the discriminator')
    parser.add_argument('--g-lr', type=float, default=0.0002, help='Learning rate for the generator')
    parser.add_argument('--nc', type=int, default=1,
                        help='Number of input channels. Ex: for grayscale images: 1 and RGB images: 3 ')
    parser.add_argument('--batch-size', type=int, default=1000, help='Batch size')
    parser.add_argument('--num-test-samples', type=int, default=16, help='Number of samples to visualize')
    parser.add_argument('--output-path', type=str, default='./results/', help='Path to save the images')
    parser.add_argument('--use-fixed', action='store_true', help='Boolean to use fixed noise or not')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ep', type=int, default=10)
    parser.add_argument('--cosine', type=int, default=0)


    opt = parser.parse_args()
    print(opt)

    ep = opt.ep
    np.random.seed(opt.seed)
    torch.random.manual_seed(opt.seed)

    current_time = datetime.now().strftime('%Y-%m-%d_%H')
    log_dir = os.path.join('runs', "DCGAN_MNIST_" + current_time, 'gamma_' + str(opt.gamma), str(ep))

    if opt.cosine:
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.0001, debias=False, cost=cosine_distance, scaling=0.99)
        log_dir = log_dir + '_cosine/'
    else:
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.0001, debias=False)
        log_dir = log_dir + '_fix/'

    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "args.txt"), "w") as fp:
        for arg in vars(opt):
            fp.write("%s:%s \n" % (arg, str(getattr(opt, arg))))

    # Gather MNIST Dataset
    train_loader = get_data_loader(opt.batch_size)
    num_batches = len(train_loader)
    fixed_noise = torch.randn(opt.num_test_samples, opt.nz, 1, 1).cuda()

    G = Generator(opt.nc, opt.nz, opt.ngf).cuda()
    D = Discriminator(opt.nc, opt.ndf).cuda()

    d_optimizer = optim.Adam(D.parameters(), lr=opt.d_lr)
    g_optimizer = optim.Adam(G.parameters(), lr=opt.g_lr)

    if opt.cosine:
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.0001, debias=False, cost=cosine_distance)
    else:
        loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.0001, debias=False)

    W1_Loss = SamplesLoss(loss="sinkhorn", p=2, blur=0.0001)

    for epoch in range(opt.num_epochs):
        generate_images(epoch, log_dir, fixed_noise, opt.num_test_samples, G, use_fixed=True)
        for i, (real_images, _) in enumerate(train_loader):
            bs = real_images.shape[0] // 2
            img = real_images.view(bs, -1)
            Dimg1, Dimg2 = real_images.cuda()[:bs, :], real_images.cuda()[bs:, :]
            images1, images2 = Dimg1.view(Dimg1.shape[0], -1), Dimg1.view(Dimg2.shape[0], -1)

            ####################
            # Build input data #
            ####################
            z_1 = torch.rand(bs, opt.nz, 1, 1).cuda()
            z_1 = Variable((z_1 - 0.5) * 2)
            z_2 = torch.rand(bs, opt.nz, 1, 1).cuda()
            z_2 = Variable((z_2 - 0.5) * 2)

            generated_imgs_1 = G.forward(z_1)
            generated_imgs_2 = G.forward(z_2)

            gamma = opt.gamma
            D_fake_1 = D.forward(generated_imgs_1)
            D_real_1 = D.forward(Dimg1)
            D_fake_1 = D_fake_1.view(D_fake_1.shape[0], -1)
            D_real_1 = D_real_1.view(D_real_1.shape[0], -1)

            D_fake_2 = D.forward(generated_imgs_2)
            D_real_2 = D.forward(Dimg2)
            D_fake_2 = D_fake_2.view(D_fake_2.shape[0], -1)
            D_real_2 = D_real_2.view(D_real_2.shape[0], -1)
            if gamma != 0 and gamma != 1:
                concat_x_1 = torch.cat((np.sqrt(gamma) * generated_imgs_1.view(bs, -1), np.sqrt(1 - gamma) * D_fake_1),
                                       1)
                concat_y_1 = torch.cat((np.sqrt(gamma) * images1, np.sqrt(1 - gamma) * D_real_1), 1)
                concat_x_2 = torch.cat((np.sqrt(gamma) * generated_imgs_2.view(bs, -1), np.sqrt(1 - gamma) * D_fake_2),
                                       1)
                concat_y_2 = torch.cat((np.sqrt(gamma) * images2, np.sqrt(1 - gamma) * D_real_2), 1)
            elif gamma == 0:
                concat_x_1 = D_fake_1
                concat_y_1 = D_real_1
                concat_x_2 = D_fake_2
                concat_y_2 = D_real_2
            else:
                concat_x_1 = generated_imgs_1.view(bs, -1)
                concat_y_1 = images1
                concat_x_2 = generated_imgs_2.view(bs, -1)
                concat_y_2 = images2

            ##########################
            # Training Discriminator #
            ##########################
            if i % ep == 0:
                for param in G.parameters():
                    param.requires_grad = False

                d_optimizer.zero_grad()

                neg_loss = -loss(concat_x_1, concat_y_1)
                neg_loss = -loss(concat_x_1, concat_y_2) + neg_loss
                neg_loss = -loss(concat_x_2, concat_y_1) + neg_loss
                neg_loss = -loss(concat_x_2, concat_y_2) + neg_loss
                neg_loss = loss(concat_x_1, concat_x_2) * 2 + neg_loss
                neg_loss = loss(concat_y_1, concat_y_2) * 2 + neg_loss

                d_loss = neg_loss
                print('D_Loss:%f' % (-d_loss.data.tolist()))
                d_loss.backward()
                d_optimizer.step()

                for param in G.parameters():
                    param.requires_grad = True

                del neg_loss
                torch.cuda.empty_cache()

            else:
                for param in D.parameters():
                    param.requires_grad = False
                pos_loss = loss(concat_x_1, concat_y_1)
                pos_loss = loss(concat_x_1, concat_y_2) + pos_loss
                pos_loss = loss(concat_x_2, concat_y_1) + pos_loss
                pos_loss = loss(concat_x_2, concat_y_2) + pos_loss
                pos_loss = -loss(concat_x_1, concat_x_2) * 2 + pos_loss
                pos_loss = -loss(concat_y_1, concat_y_2) * 2 + pos_loss

                g_loss = pos_loss
                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()
                for param in D.parameters():
                    param.requires_grad = True

        if epoch % 20 == 0 and epoch != 0:
            torch.save(G.state_dict(), os.path.join(log_dir, "gauss_iter_%i_G" % epoch))
            torch.save(D.state_dict(), os.path.join(log_dir, "gauss_iter_%i_D" % epoch))



import torch.nn as nn
import numpy as np
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self, input_channels, n_feature_maps):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(input_channels, n_feature_maps, 4, 2, 1, bias=False),
            nn.ELU(),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(n_feature_maps, n_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 2),
            nn.ELU(),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(n_feature_maps * 2, n_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 4),
            nn.ELU(),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(n_feature_maps * 4, n_feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 8),
            nn.ELU(),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(n_feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_feature_maps * 16),
            nn.ELU()
        )

    def forward(self, input):
        output = self.main(input)
        output = output.view([-1,np.prod(output.shape[1:])])
        output = output/torch.norm(output,dim=1,keepdim=True)
        return output


class Generator(nn.Module):
    def __init__(self, input_channels, n_feature_maps):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(input_channels, n_feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(n_feature_maps * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(n_feature_maps * 8, n_feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(n_feature_maps * 4, n_feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(n_feature_maps * 2, n_feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(n_feature_maps),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(n_feature_maps, input_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)

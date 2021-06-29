import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(nz, 4096)
        self.ELU = nn.ELU()
        self.network = nn.Sequential(
            nn.BatchNorm2d(4096),
            nn.ELU(),
            nn.ConvTranspose2d(4096, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ELU(),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ELU(),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ELU(),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        out = self.fc1(input.view(input.shape[0],-1))
        out = self.ELU(out)
        out = out.view(input.shape[0],4096,1,1)
        out = self.network(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.network = nn.Sequential(

            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.ELU(),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.ELU(),

            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.ELU()
        )
        self.fc = nn.Linear(4096,1024)

    def forward(self, input):
        output = self.network(input)
        output = output.view([-1,np.prod(output.shape[1:])])
        output = self.fc(output)
        output = output/torch.norm(output,dim=1,keepdim=True)
        return output
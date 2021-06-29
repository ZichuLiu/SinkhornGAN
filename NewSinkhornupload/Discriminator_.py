import torch
import torch.nn as nn
from torch.autograd import Variable, grad
import torch.optim as optim
from torch.distributions.normal import Normal

DEVICE = 0


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=6):
        super(Discriminator, self).__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        for i in range(num_layers - 2):
            layers.extend([nn.ELU(),
                           nn.Linear(hidden_size, hidden_size)])
        layers.append(nn.Linear(hidden_size, output_size).cuda(device=DEVICE))
        self.net = nn.Sequential(*layers)
        self.out = nn.LayerNorm(output_size)

    def forward(self, x):
        return self.out(self.net(x))

    def get_penalty(self, x_true, x_gen):
        x_true = x_true.view_as(x_gen).cuda()
        alpha = torch.rand((len(x_true),) + (1,) * (x_true.dim() - 1))
        if x_true.is_cuda:
            alpha = alpha.cuda(x_true.get_device())
        x_penalty = Variable(alpha * x_true + (1 - alpha) * x_gen, requires_grad=True).cuda()
        p_penalty = self.forward(x_penalty)
        gradients = grad(p_penalty, x_penalty, grad_outputs=torch.ones_like(p_penalty).cuda(
            x_true.get_device()) if x_true.is_cuda else torch.ones_like(p_penalty), create_graph=True,
                         retain_graph=True, only_inputs=True)[0]
        penalty = ((gradients.view(len(x_true), -1).norm(2, 1) - 1) ** 2).mean()

        return penalty

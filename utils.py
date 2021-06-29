import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import math
import itertools
# import imageio
# import natsort
from glob import glob


def get_data_loader(batch_size):
    # MNIST Dataset
    transform = transforms.Compose([
        transforms.ToTensor()])

    train_dataset = datasets.MNIST(root='/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/data/', train=True, transform=transform, download=True)

    # Data Loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def generate_images(epoch, path, fixed_noise, num_test_samples, netG, use_fixed=False):
    z = torch.randn(num_test_samples, fixed_noise.shape[0], 1, 1).cuda()
    size_figure_grid = int(math.sqrt(num_test_samples))
    title = None

    if use_fixed:
        generated_fake_images = netG(fixed_noise)
    else:
        generated_fake_images = netG(z)

    fig, ax = plt.subplots(size_figure_grid, size_figure_grid, figsize=(6, 6))
    for i, j in itertools.product(range(size_figure_grid), range(size_figure_grid)):
        ax[i, j].get_xaxis().set_visible(False)
        ax[i, j].get_yaxis().set_visible(False)
    for k in range(num_test_samples):
        i = k // 4
        j = k % 4
        ax[i, j].cla()
        ax[i, j].imshow(generated_fake_images[k].data.cpu().numpy().reshape(28, 28), cmap='Greys')
    label = 'Epoch_{}'.format(epoch + 1)
    fig.text(0.5, 0.04, label, ha='center')
    fig.suptitle(title)
    fig.savefig(path + label + '.png')
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from DCGAN import Discriminator

# dataset_name = "LSUN"
dataset_name = "MNIST"
batch_size = 4000
if dataset_name == "LSUN":
    total_epoch = 100000
    img_size = 64
    image_chanel = 3
    epsilon = 10
    root = '/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/data/lsun-master/'
    trans = transforms.Compose([transforms.Resize(img_size), transforms.CenterCrop(img_size), transforms.ToTensor,
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])
    data_set = dset.LSUN(root=root, classes=['bedroom_train'], transform=trans)

if dataset_name=="MNIST":
    total_epoch = 100000
    img_size = 28
    image_chanel = 1
    epsilon = 10
    root = '/scratch/gobi2/jeanlancel/SinkhornAutoDiff-master/data/'
    data_set = dset.MNIST(root=root,train=True,download=True)
    test_set = dset.MNIST(root=root,train=False)

data_loader = torch.utils.data.DataLoader(dataset=data_set,batch_size = batch_size,shuffle=True)
D = Discriminator(input_channels=image_chanel,n_feature_maps=64)
data_iter = iter(data_loader)
data = data_iter.next()[0]
print(D(data))
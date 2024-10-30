from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from six.moves import xrange

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

from modules import Model


def show(img):
    npimg = img.numpy()
    fig = plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)


def train(model, num_training_updates, training_loader, optimizer, data_variance, device):
    model.train()
    train_res_recon_error = []

    for i in xrange(num_training_updates):
        (data, _) = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad()

        vq_loss, data_recon = model(data)
        recon_error = F.mse_loss(data_recon, data) / data_variance
        loss = recon_error + vq_loss
        loss.backward()

        optimizer.step()

        train_res_recon_error.append(recon_error.item())

        if (i + 1) % 100 == 0:
            print('%d iterations' % (i + 1))
            print('recon_error: %.3f' % np.mean(train_res_recon_error[-100:]))
            print()


def val(model, validation_loader, device):
    model.eval()

    (valid_originals, _) = next(iter(validation_loader))
    valid_originals = valid_originals.to(device)

    vq_output_eval = model._pre_vq_conv(model._encoder(valid_originals))
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)

    show(make_grid(valid_reconstructions.cpu().data) + 0.5, )
    show(make_grid(valid_originals.cpu() + 0.5))


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = 256
    num_training_updates = 15000
    num_hiddens = 128
    num_residual_hiddens = 32
    num_residual_layers = 2
    embedding_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25
    learning_rate = 1e-3

    training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                     ]))

    validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                       transform=transforms.Compose([
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (1.0, 1.0, 1.0))
                                       ]))

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=True, pin_memory=True)

    model = Model(num_hiddens, num_residual_layers, num_residual_hiddens, num_embeddings, embedding_dim,
                  commitment_cost).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    data_variance = np.var(training_data.data / 255.0)
    train(model, num_training_updates, training_loader, optimizer, data_variance, device)
    val(model, validation_loader, device)
# Based on https://debuggercafe.com/autoencoder-neural-network-application-to-image-denoising/
import numpy as np
import torch.nn.functional as F
from torch import nn
import torch


class Autoencoder(nn.Module):
    def __init__(self, args):
        super(Autoencoder, self).__init__()
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        # encoder layers
        # self.enc1 = nn.Conv3d(1, 128, kernel_size=(3, 3, 3), padding=1).to(self.device)
        self.enc2 = nn.Conv3d(1, 64, kernel_size=(3, 3, 3), padding=1).to(self.device)
        self.enc3 = nn.Conv3d(64, 32, kernel_size=(3, 3, 3), padding=1).to(self.device)
        # self.enc4 = nn.Conv3d(32, 16, kernel_size=(3, 3, 3), padding=1).to(self.device)
        # self.enc5 = nn.Conv3d(16, 8, kernel_size=(3, 3, 3), padding=1).to(self.device)
        self.pool = nn.MaxPool3d(2, 2).to(self.device)

        # decoder layers
        # self.dec1 = nn.ConvTranspose3d(8, 8, kernel_size=2, stride=(2, 2, 2)).to(self.device)
        # self.dec2 = nn.ConvTranspose3d(16, 16, kernel_size=(2, 2, 2), stride=(2, 2, 2)).to(self.device)
        self.dec3 = nn.ConvTranspose3d(32, 32, kernel_size=(2, 2, 2), stride=(2, 2, 2)).to(self.device)
        self.dec4 = nn.ConvTranspose3d(32, 64, kernel_size=(2, 2, 2), stride=(2, 2, 2)).to(self.device)
        # self.dec5 = nn.ConvTranspose3d(64, 128, kernel_size=(3, 3, 3), stride=2).to(self.device)
        self.out = nn.Conv3d(64, 1, kernel_size=(3, 3, 3), padding=1).to(self.device)

    def forward(self, x):
        # encode
        # x = F.relu(self.enc1(x))
        # x = self.pool(x)
        if not torch.is_tensor(x):
            x = x[:,:,:,:,0]
            x = x[np.newaxis,:,:,:,:]
            x = torch.Tensor(x).to(self.device)
        x = F.relu(self.enc2(x))
        x = self.pool(x)
        x = F.relu(self.enc3(x))
        # x = self.pool(x)
        # x = F.relu(self.enc4(x))
        # x = self.pool(x)
        # x = F.relu(self.enc5(x))
        x = self.pool(x)  # the latent space representation

        # decode
        # x = F.relu(self.dec1(x))
        # x = F.relu(self.dec2(x))
        x = F.relu(self.dec3(x))
        x = F.relu(self.dec4(x))
        # x = F.relu(self.dec5(x))
        x = torch.sigmoid(self.out(x))
        return x

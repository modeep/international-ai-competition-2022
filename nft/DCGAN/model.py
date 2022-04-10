import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,
                 z_dim=100,
                 im_chan=3,
                 hidden_dim=64):

        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.im_chan = im_chan
        self.hidden_dim = hidden_dim

        self.generator_cnn = nn.Sequential(self.block(z_dim, hidden_dim * 8, stride=1, padding=0),

                                           self.block(hidden_dim * 8, hidden_dim * 4),

                                           self.block(hidden_dim * 4, hidden_dim * 2),

                                           self.block(hidden_dim * 2, hidden_dim),

                                           self.block(hidden_dim, im_chan, final_layer=True))

    def block(self,
                       im_chan,
                       op_chan,
                       kernel_size=4,
                       stride=2,
                       padding=1,
                       final_layer=False):

        layers = []

        layers.append(nn.ConvTranspose2d(im_chan,
                                         op_chan,
                                         kernel_size,
                                         stride,
                                         padding,
                                         bias=False))

        if not final_layer:
            layers.append(nn.BatchNorm2d(op_chan))
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def forward(self, noise):
        x = noise.view(-1, self.z_dim, 1, 1)
        return self.generator_cnn(x)

    def get_noise(n_samples,
                  z_dim,
                  device='cpu'):
        return torch.randn(n_samples,
                           z_dim,
                           device=device)


class Discriminator(nn.Module):
    def __init__(self,
                 im_chan=3,
                 conv_dim=64,
                 image_size=64):

        super(Discriminator, self).__init__()
        self.image_size = image_size
        self.conv_dim = conv_dim

        self.disc_cnn = nn.Sequential(self.block(im_chan, conv_dim),
                                      self.block(conv_dim, conv_dim * 2),
                                      self.block(conv_dim * 2, conv_dim * 4),
                                      self.block(conv_dim * 4, conv_dim * 8),

                                      self.block(conv_dim * 8, 1, padding=0, final_layer=True))

    def block(self,
                        im_chan,
                        op_chan,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                        final_layer=False):
        layers = []
        layers.append(nn.Conv2d(im_chan,
                                op_chan,
                                kernel_size,
                                stride,
                                padding,
                                bias=False))

        if not final_layer:
            layers.append(nn.BatchNorm2d(op_chan))
            layers.append(nn.LeakyReLU(0.2, inplace=True))

        return nn.Sequential(*layers)

    def forward(self, image):
        pred = self.disc_cnn(image)
        pred = pred.view(image.size(0), -1)
        return pred

    def _get_final_feature_dimention(self):
        final_width_height = (self.image_size // 2 ** len(self.disc_cnn)) ** 2
        final_depth = self.conv_dim * 2 ** (len(self.disc_cnn) - 1)
        return final_depth * final_width_height
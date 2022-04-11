import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
from model import Discriminator, Generator
from dataloader import get_dataloader
from loss import fake_loss, real_loss
import os
from tqdm import tqdm
os.environ['KMP_DUPLICATE_LIB_OK']='True'

image_dir = "datasets/"
image_root = "datasets/"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device is ", device)

generator = Generator(z_dim=100,
              im_chan=3,
              hidden_dim=64).to(device)

discriminator = Discriminator(im_chan=3,
                              conv_dim=64,
                              image_size=64).to(device)

def save_images(images_tensor, epoch):

    plt.rcParams['figure.figsize'] = (15, 15)
    plt.subplots_adjust(wspace=0, hspace=0)

    images_tensor = images_tensor.to('cpu')
    npimgs = images_tensor.detach().numpy()

    no_plots = len(images_tensor)

    for idx, image in enumerate(npimgs):
        plt.subplot(1, 8, idx + 1)
        plt.axis('off')

        image = image * 0.5 + 0.5
        image = np.transpose(image, (1, 2, 0))
        plt.imshow(image)

    plt.savefig(f'{epoch}.png')

def train(D, G,
          start_epochs,
          n_epochs,
          dataloader,
          d_optimizer,
          g_optimizer,
          z_dim,
          save_image_every=10,
          save_model_every=10,
          device='cpu'):
    sample_size = 8
    fixed_z = Generator.get_noise(n_samples=sample_size,
                                  z_dim=z_dim,
                                  device=device)

    for epoch in tqdm(range(start_epochs + 1, n_epochs + 1)):
        for batch_i, (real_images, _) in tqdm(enumerate(dataloader)):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)

            d_optimizer.zero_grad()

            d_real_op = D(real_images)
            d_real_loss = real_loss(d_real_op,
                                    device=device)

            noise = Generator.get_noise(n_samples=batch_size,
                                        z_dim=z_dim,
                                        device=device)
            fake_images = G(noise)

            d_fake_op = D(fake_images)
            d_fake_loss = fake_loss(d_fake_op,
                                    device=device)

            d_loss = d_real_loss + d_fake_loss

            d_loss.backward()

            d_optimizer.step()

            g_optimizer.zero_grad()
            noise = Generator.get_noise(n_samples=batch_size,
                                        z_dim=z_dim,
                                        device=device)

            g_out = G(noise)
            d_out = D(g_out)

            g_loss = real_loss(d_out,
                               device=device)

            g_loss.backward()

            g_optimizer.step()

        print('Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(epoch,
                                                                               n_epochs,
                                                                               d_loss.item(),
                                                                               g_loss.item()))
        if (epoch % save_image_every == 0):
            G.eval()
            sample_image = G(fixed_z)
            save_images(sample_image, epoch)
            G.train()

        if (epoch % save_model_every == 0):
            torch.save({
                'epoch': epoch,
                'G_optimizer':g_optimizer.state_dict(),
                'D_optimizer':d_optimizer.state_dict(),
                'Generator':G.state_dict(),
                'Discriminator':D.state_dict()
            }, f"weights/{epoch}_model.tar")

checkpoint = torch.load("trained_weight/500_model.tar")
generator.load_state_dict(checkpoint["Generator"])
discriminator.load_state_dict(checkpoint["Discriminator"])

dataloader = get_dataloader(128,
                            64,
                            image_root)

beta_1 = 0.5
beta_2 = 0.999
lr = 0.0002

g_optimizer = optim.Adam(generator.parameters(),
                         lr=lr,
                         betas=(beta_1, beta_2))

d_optimizer = optim.Adam(discriminator.parameters(),
                         lr=lr,
                         betas=(beta_1, beta_2))

g_optimizer.load_state_dict(checkpoint["G_optimizer"])
d_optimizer.load_state_dict(checkpoint["D_optimizer"])

n_epochs = 500
train(discriminator,
      generator,
      500,
      n_epochs,
      dataloader,
      d_optimizer,
      g_optimizer,
      100,
      save_image_every=1,
      save_model_every=10,
      device=device)
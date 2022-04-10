from model import Generator
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
os.environ['KMP_DUPLICATE_LIB_OK']= 'True'

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = Generator(z_dim=100,
              im_chan=3,
              hidden_dim=64)

sample_size = 8
z_dim = 100

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

    plt.savefig(f'result/{epoch}.png')

checkpoint = torch.load("weights/450_model.tar")
generator.load_state_dict(checkpoint["Generator"])
generator.to(device)
generator.eval()
for i in range(0, 10):
    fixed_z = Generator.get_noise(n_samples=sample_size,
                                      z_dim=z_dim,
                                      device=device)
    sample_image = generator(fixed_z)
    save_images(sample_image, i)
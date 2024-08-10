import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from models.generator import Generator
from models.discriminator import Discriminator
from utils.data_loader import get_dataloader


def train_sagan(generator, discriminator, dataloader, num_epochs, z_dim, device):
    criterion = nn.BCEWithLogitsLoss()
    optimizerG = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizerD = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    fixed_noise = torch.randn(64, z_dim, 1, 1, device=device)

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            discriminator.zero_grad()
            real_data = data[0].to(device)
            batch_size = real_data.size(0)
            labels_real = torch.ones(batch_size, device=device)
            labels_fake = torch.zeros(batch_size, device=device)
            output = discriminator(real_data)
            errD_real = criterion(output, labels_real)
            errD_real.backward()
            noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
            fake_data = generator(noise)
            output = discriminator(fake_data.detach())
            errD_fake = criterion(output, labels_fake)
            errD_fake.backward()
            optimizerD.step()
            generator.zero_grad()
            output = discriminator(fake_data)
            errG = criterion(output, labels_real)
            errG.backward()
            optimizerG.step()
            if i % 50 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}] Batch {i}/{len(dataloader)} \
                       Loss D: {errD_real.item() + errD_fake.item()}, loss G: {errG.item()}')
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                fake_images = generator(fixed_noise).detach().cpu()
            save_image(fake_images, f'outputs/fake_images_epoch_{epoch+1}.png', normalize=True)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z_dim = 100
    image_channels = 3

    generator = Generator(z_dim, image_channels).to(device)
    discriminator = Discriminator(image_channels).to(device)

    dataloader = get_dataloader(batch_size=64)
    train_sagan(generator, discriminator, dataloader, num_epochs=100, z_dim=z_dim, device=device)

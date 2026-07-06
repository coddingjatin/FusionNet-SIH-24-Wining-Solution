import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset Class for SAR and Optical Images
class SAROpticalDataset(Dataset):
    def _init_(self, sar_dir, optical_dir, transform=None):
        self.sar_dir = sar_dir
        self.optical_dir = optical_dir
        self.sar_images = os.listdir(sar_dir)
        self.optical_images = os.listdir(optical_dir)
        self.transform = transform

    def _len_(self):
        return len(self.sar_images)

    def _getitem_(self, idx):
        # Load SAR image
        sar_img_path = os.path.join(self.sar_dir, self.sar_images[idx])
        sar_image = Image.open(sar_img_path).convert("L")

        # Load corresponding Optical image
        optical_img_path = os.path.join(self.optical_dir, self.optical_images[idx])
        optical_image = Image.open(optical_img_path).convert("RGB")

        # Apply transformations
        if self.transform:
            sar_image = self.transform(sar_image)
            optical_image = self.transform(optical_image)

        return sar_image, optical_image

# Image Transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Dataset Directories
sar_dir = r"D:\SIH\sentinel_dataset\train\sar"
optical_dir = r"D:\SIH\sentinel_dataset\train\optical"

# Create Dataset and DataLoader
dataset = SAROpticalDataset(sar_dir=sar_dir, optical_dir=optical_dir, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)


# Attention Mechanism
class AttentionBlock(nn.Module):
    def _init_(self, in_channels):
        super(AttentionBlock, self)._init_()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        energy = torch.bmm(query.permute(0, 2, 1), key)  # BxNqxNk
        attention = torch.softmax(energy, dim=-1)  # BxNq x Nk

        out = torch.bmm(value, attention.permute(0, 2, 1))  # BxNc x Nq
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out

# Encoder and Decoder Blocks
class Encoder(nn.Module):
    def _init_(self, input_nc):
        super(Encoder, self)._init_()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def _init_(self, output_nc):
        super(Decoder, self)._init_()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_nc, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)


# Generator Network with Encoder-Decoder Architecture
class Generator(nn.Module):
    def _init_(self, input_nc, output_nc):
        super(Generator, self)._init_()
        self.encoder = Encoder(input_nc)
        self.attention = AttentionBlock(256)
        self.decoder = Decoder(output_nc)

    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)
        x = self.decoder(x)
        return x

# Discriminator Network
class Discriminator(nn.Module):
    def _init_(self, input_nc):
        super(Discriminator, self)._init_()
        self.model = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

# Initialize Generators and Discriminators
G1 = Generator(input_nc=1, output_nc=3).to(device)
G2 = Generator(input_nc=3, output_nc=1).to(device)
D1 = Discriminator(input_nc=3).to(device)
D2 = Discriminator(input_nc=1).to(device)

# Define Loss Functions and Optimizers
criterion_GAN = nn.BCELoss()
criterion_cycle = nn.L1Loss()
criterion_identity = nn.L1Loss()

optimizer_G1 = optim.Adam(G1.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_G2 = optim.Adam(G2.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D1 = optim.Adam(D1.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D2 = optim.Adam(D2.parameters(), lr=0.0002, betas=(0.5, 0.999))


# Initialize best training losses
best_train_loss_G1 = float("inf")
best_train_loss_G2 = float("inf")

# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    G1.train()
    G2.train()
    D1.train()
    D2.train()

    epoch_loss_G1 = 0
    epoch_loss_G2 = 0

    for i, (sar_images, optical_images) in enumerate(dataloader):
        sar_images = sar_images.to(device)
        optical_images = optical_images.to(device)

        optimizer_G1.zero_grad()
        optimizer_G2.zero_grad()
        optimizer_D1.zero_grad()
        optimizer_D2.zero_grad()

        fake_optical = G1(sar_images)
        reconstructed_sar = G2(fake_optical)

        fake_sar = G2(optical_images)
        reconstructed_optical = G1(fake_sar)

        # Calculate losses
        loss_GAN_G1 = criterion_GAN(D1(fake_optical), torch.ones_like(D1(fake_optical)))
        loss_GAN_G2 = criterion_GAN(D2(fake_sar), torch.ones_like(D2(fake_sar)))
        loss_cycle_SAR = criterion_cycle(reconstructed_sar, sar_images) * 10.0
        loss_cycle_Optical = criterion_cycle(reconstructed_optical, optical_images) * 10.0

        # Total generator loss
        loss_G1 = loss_GAN_G1 + loss_cycle_SAR
        loss_G2 = loss_GAN_G2 + loss_cycle_Optical

        # Accumulate losses for the epoch
        epoch_loss_G1 += loss_G1.item()
        epoch_loss_G2 += loss_G2.item()

        # Backpropagation
        loss_G1.backward()
        loss_G2.backward()
        optimizer_G1.step()
        optimizer_G2.step()

        # Discriminator Losses
        loss_D1_real = criterion_GAN(D1(optical_images), torch.ones_like(D1(optical_images)))
        loss_D1_fake = criterion_GAN(D1(fake_optical.detach()), torch.zeros_like(D1(fake_optical)))
        loss_D2_real = criterion_GAN(D2(sar_images), torch.ones_like(D2(sar_images)))
        loss_D2_fake = criterion_GAN(D2(fake_sar.detach()), torch.zeros_like(D2(fake_sar)))

        loss_D1 = (loss_D1_real + loss_D1_fake) * 0.5
        loss_D2 = (loss_D2_real + loss_D2_fake) * 0.5

        loss_D1.backward()
        loss_D2.backward()
        optimizer_D1.step()
        optimizer_D2.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], "
              f"Loss_G1: {loss_G1.item():.4f}, Loss_G2: {loss_G2.item():.4f}, "
              f"Loss_D1: {loss_D1.item():.4f}, Loss_D2: {loss_D2.item():.4f}")

    # Average training losses for the epoch
    epoch_loss_G1 /= len(dataloader)
    epoch_loss_G2 /= len(dataloader)

    print(f"Average Training Loss for G1 after Epoch [{epoch+1}/{num_epochs}]: {epoch_loss_G1:.4f}")
    print(f"Average Training Loss for G2 after Epoch [{epoch+1}/{num_epochs}]: {epoch_loss_G2:.4f}")

    # Save the best models based on training loss
    if epoch_loss_G1 < best_train_loss_G1:
        best_train_loss_G1 = epoch_loss_G1
        torch.save(G1.state_dict(), "sentinel_G1_model.pth")
        print(f"Best G1 model saved with training loss: {best_train_loss_G1:.4f}")

    if epoch_loss_G2 < best_train_loss_G2:
        best_train_loss_G2 = epoch_loss_G2
        torch.save(G2.state_dict(), "sentinel_G2_model.pth")
        print(f"Best G2 model saved with training loss: {best_train_loss_G2:.4f}")



# Load the models
G1 = Generator(input_nc=1, output_nc=3).to(device)
G2 = Generator(input_nc=3, output_nc=1).to(device)

G1.load_state_dict(torch.load("../weigths/sentinel_G1_model.pth"))
G2.load_state_dict(torch.load("../weigths/sentinel_G2_model.pth"))

G1.eval()
G2.eval()


from PIL import Image
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load and preprocess the grayscale image
test_image_path = r"path/to/SAR/image"
sar_image = Image.open(test_image_path).convert("L")
sar_image = transform(sar_image).unsqueeze(0).to(device)



with torch.no_grad():
    generated_optical_image = G1(sar_image)


from torchvision.utils import save_image

# Denormalize and save the image
generated_optical_image = generated_optical_image * 0.5 + 0.5
save_image(generated_optical_image, "generated_optical_image.png")
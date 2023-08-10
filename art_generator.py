import torch
import torch.nn as nn
import torch.optim as optim
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 28 * 28 * 3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  
        )

    def forward(self, x):
        return self.model(x)

LATENT_DIM = 100
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0002
BETA1 = 0.5

generator = Generator(LATENT_DIM)
discriminator = Discriminator()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

generator_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
loss_fn = nn.BCELoss()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageFolder(root="/home/sajad/Downloads/dataset", transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    print(f"Epoch [{epoch+1}/{EPOCHS}]")
    for batch_images, _ in dataloader:
        real_images = batch_images.to(device)
        batch_size = real_images.size(0)
        
       
        discriminator_optimizer.zero_grad()
        noise = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_images = generator(noise)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)
        real_loss = loss_fn(discriminator(real_images), real_labels)
        fake_loss = loss_fn(discriminator(fake_images.detach()), fake_labels)
        disc_loss = real_loss + fake_loss
        disc_loss.backward()
        discriminator_optimizer.step()

        generator_optimizer.zero_grad()
        noise = torch.randn(batch_size, LATENT_DIM).to(device)
        fake_images = generator(noise)
        gen_loss = loss_fn(discriminator(fake_images), real_labels)
        gen_loss.backward()
        generator_optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch [{epoch}/{EPOCHS}] - Saving generated images...")
        # Save generated images
        generated_images = generator(torch.randn(BATCH_SIZE, LATENT_DIM).to(device))
        generated_images = (generated_images + 1) / 2  


        os.makedirs("generated_images", exist_ok=True)

        for i in range(BATCH_SIZE):
            generated_image = generated_images[i].cpu() 
            generated_image = generated_image.view(3, 28, 28)  
            print("Generated image shape:", generated_image.shape)  
            image = transforms.ToPILImage()(generated_image)
            image.save(f"generated_images/epoch_{epoch}_sample_{i}.png")

        print(f"Epoch {epoch}/{EPOCHS} - Saved generated images.")


torch.save(generator.state_dict(), "generator_model.pth")
print("Training complete. Generator model saved.")



torch.save(generator.state_dict(), "generator_model.pth")
print("Training complete. Generator model saved.")
print("Current working directory:", os.getcwd())



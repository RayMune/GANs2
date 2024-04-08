import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from PIL import Image

# Define the transformation to apply to the images
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Resize the images to 28x28 pixels
    transforms.ToTensor(),  # Convert the images to PyTorch tensors
    # Add any other transformations you want to apply to the images here
])

# Define a custom image loader that loads images from a flat folder structure
path='C:\\GANs\\images1'
def custom_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

# Create the dataset from the images in the 'images' folder
dataset = ImageFolder(root='C:/GANs', transform=transform, loader=custom_loader)

# Create the dataloader that loads batches of images from the dataset
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Generate and save an image
with torch.no_grad():
    noise = torch.randn(1, 100)
    fake_image = generator(noise)
    fake_image = fake_image.detach().cpu().numpy()  # Convert to NumPy array
    fake_image = fake_image.reshape(28, 28)  # Reshape to 28x28 pixels
    save_image(torch.tensor(fake_image), 'generated_image.png')

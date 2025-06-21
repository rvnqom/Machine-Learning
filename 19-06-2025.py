import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5),(0.5))
])
train_dataset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

train_loader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=8,
    shuffle=True
)
classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
           'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
data_iter = iter(train_loader)
images, labels = next(data_iter)
images = images * 0.5 + 0.5
fig, axes = plt.subplots(1, 8, figsize=(14, 2))
for idx in range(8):
    axes[idx].imshow(images[idx].squeeze(), cmap='gray')
    axes[idx].set_title(classes[labels[idx]])
    axes[idx].axis('off')

plt.suptitle('Sample Fashion MNIST Images', fontsize=14)
plt.show()

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(batch_size, image_size=64, data_dir='./data'):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = datasets.CelebA(root=data_dir, split='train', transform=transform, download=False)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    dataset = datasets.CelebA(root='./data', split='train', transform=transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

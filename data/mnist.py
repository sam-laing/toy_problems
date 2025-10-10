import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

def get_mnist_dataloaders(batch_size=64, val_ratio=0.1) -> tuple[DataLoader, DataLoader, DataLoader]:
    """   
    Gets a train, validation, and test DataLoader for the MNIST dataset.
    Args:
        batch_size (int): Number of samples per batch.
        val_ratio (float): Proportion of training data to use for validation.
        download (bool): Whether to download the dataset if not present.
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    data_path = "/fast/slaing/data/vision/mnist/MNIST/"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.Lambda(lambda x: x.view(-1)) 
    ])

    train_dataset = datasets.MNIST(data_path, train=True, transform=transform)
    test_dataset = datasets.MNIST(data_path, train=False, transform=transform)

    val_size = int(len(train_dataset) * val_ratio)
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    print("something is actually happening")
    train_loader, val_loader, test_loader = get_mnist_dataloaders(batch_size=64, val_ratio=0.1)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")




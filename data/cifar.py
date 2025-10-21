import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

def get_cifar_dataloaders(batch_size=64, val_ratio=0.1, seed=42, download=False) -> tuple[DataLoader, DataLoader, DataLoader]:
    
    """   
    Gets a train, validation, and test DataLoader for the CIFAR-10 dataset.
    Args:
        batch_size (int): Number of samples per batch.
        val_ratio (float): Proportion of training data to use for validation.
        seed (int): Random seed for reproducibility.
        download (bool): Whether to download the dataset if not present.
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    if seed is not None:
        torch.manual_seed(seed)
        
    data_path = "/fast/slaing/data/vision/cifar10/"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Lambda(lambda x: x.view(-1)) 
    ])

    train_dataset = datasets.CIFAR10(data_path, train=True, transform=transform, download=download)
    test_dataset = datasets.CIFAR10(data_path, train=False, transform=transform, download=download)

    #stratified split: same number of each class in train and val
    if val_ratio > 0:
        targets = torch.tensor(train_dataset.targets)
        classes = torch.unique(targets)
        val_indices = []
        train_indices = []
        
        for cls in classes:
            cls_indices = torch.where(targets == cls)[0]
            val_count = int(len(cls_indices) * val_ratio)
            if seed is not None:
                generator = torch.Generator().manual_seed(seed + int(cls.item()))
                perm = torch.randperm(len(cls_indices), generator=generator)
            else:
                perm = torch.randperm(len(cls_indices))
            
            cls_val_indices = cls_indices[perm[:val_count]]
            cls_train_indices = cls_indices[perm[val_count:]]
            val_indices.append(cls_val_indices)
            train_indices.append(cls_train_indices)
        
        val_indices, train_indices = torch.cat(val_indices), torch.cat(train_indices)
        train_dataset, val_dataset = Subset(train_dataset, train_indices), Subset(train_dataset, val_indices)
    else:
        val_dataset = Subset(train_dataset, [])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader
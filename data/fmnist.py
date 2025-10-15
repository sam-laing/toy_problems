import torch 
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split, Subset

#duplicate above but for fashion mnist
def get_fmnist_dataloaders(batch_size=64, val_ratio=0.1, seed=42, download=True) -> tuple[DataLoader, DataLoader, DataLoader]:
    """   
    Gets a train, validation, and test DataLoader for the FashionMNIST dataset.
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
        
    data_path = "/fast/slaing/data/vision/fashion_mnist/"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
        transforms.Lambda(lambda x: x.view(-1)) 
    ])

    train_dataset = datasets.FashionMNIST(data_path, train=True, transform=transform, download=download)
    test_dataset = datasets.FashionMNIST(data_path, train=False, transform=transform, download=download)

    #stratified split: same number of each class in train and val
    if val_ratio > 0:
        targets = train_dataset.targets.cpu()
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

if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_fmnist_dataloaders(batch_size=64, val_ratio=0.1, seed=42)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    #print a batch
    data_iter = iter(train_loader)
    images, labels = next(data_iter)
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")
    print(f"First 10 labels: {labels[:10]}")
    print(f"First image tensor: {images[0]}")
    #check if stratified split worked
    if hasattr(train_loader.dataset, "dataset") and hasattr(train_loader.dataset.dataset, "targets"):
        train_indices = train_loader.dataset.indices
        val_indices = val_loader.dataset.indices
        all_targets = train_loader.dataset.dataset.targets


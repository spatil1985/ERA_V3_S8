import torch
from torchvision import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os

class CIFAR10Dataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform
        
    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = np.array(image)
            
        if self.transform:
            image = self.transform(image=image)["image"]
            
        return image, label
    
    def __len__(self):
        return len(self.dataset)

def save_sample_grid(dataset, transforms=None, prefix=""):
    """Save a grid of sample images"""
    if not os.path.exists('sample_images'):
        os.makedirs('sample_images')
        
    # Get class names
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle(f'CIFAR10 Images: {prefix}', fontsize=16)
    
    # Get 9 random samples
    indices = np.random.choice(len(dataset), 9, replace=False)
    
    for idx, ax in zip(indices, axes.flatten()):
        image, label = dataset.dataset[idx]
        image_np = np.array(image)
        
        # Show original image
        ax.imshow(image_np)
        ax.axis('off')
        ax.set_title(f'Class: {classes[label]}')
        
        # If transforms exist, show transformed image as inset
        if transforms is not None:
            try:
                # Apply transform
                transformed = transforms(image=image_np)['image']
                if isinstance(transformed, torch.Tensor):
                    transformed = transformed.numpy().transpose(1, 2, 0)
                transformed = np.clip(transformed, 0, 1)
                
                # Create inset axes for transformed image
                inset_ax = ax.inset_axes([0.65, 0.65, 0.35, 0.35])
                inset_ax.imshow(transformed)
                inset_ax.axis('off')
                inset_ax.set_title('Augmented', fontsize=8)
            except Exception as e:
                print(f"Warning: Could not apply transforms: {e}")
    
    plt.tight_layout()
    filename = f'sample_images/cifar10_grid_{prefix}.png'
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f'Saved {filename}')

def get_data_loaders(batch_size, apply_transforms=True):
    """Get train and test data loaders"""
    # CIFAR10 mean and std
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    
    # Define transforms
    train_transforms = None
    test_transforms = None
    
    if apply_transforms:
        train_transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(
                max_holes=1,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=[x * 255.0 for x in mean],
                p=0.5,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
        
        test_transforms = A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    # Get datasets
    train_dataset = datasets.CIFAR10('./data', train=True, download=True)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True)
    
    # Wrap with custom dataset
    train_dataset = CIFAR10Dataset(train_dataset, train_transforms)
    test_dataset = CIFAR10Dataset(test_dataset, test_transforms)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, test_loader 
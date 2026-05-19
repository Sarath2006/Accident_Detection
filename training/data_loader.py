"""
Data loader for accident detection with image datasets and CSV labels
Supports both accident/non-accident folder structure and CSV-based labels
"""
import os
import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Tuple, Optional, List


class AccidentImageDataset(Dataset):
    """
    Dataset for loading accident/non-accident images from folder structure
    
    Folder structure:
    datasets/
    ├── Accident/        -> label=1
    │   ├── 1.jpg
    │   ├── 2.jpg
    │   └── ...
    └── NonAccident/     -> label=0
        ├── 1.jpg
        ├── 2.jpg
        └── ...
    """
    
    def __init__(
        self,
        root_dir: str,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize accident dataset from folder structure
        
        Args:
            root_dir: Root directory containing Accident/ and NonAccident/ folders
            transform: Torchvision transforms to apply
            target_size: Target image size (default: 224x224 for ResNet50)
        """
        self.root_dir = root_dir
        self.target_size = target_size
        self.images = []
        self.labels = []
        
        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        # Load Accident images (label=1)
        accident_dir = os.path.join(root_dir, 'Accident')
        if os.path.exists(accident_dir):
            for img_file in os.listdir(accident_dir):
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(accident_dir, img_file)
                    self.images.append(img_path)
                    self.labels.append(1)  # Accident
        
        # Load NonAccident images (label=0)
        non_accident_dir = os.path.join(root_dir, 'NonAccident')
        if os.path.exists(non_accident_dir):
            for img_file in os.listdir(non_accident_dir):
                if img_file.lower().endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(non_accident_dir, img_file)
                    self.images.append(img_path)
                    self.labels.append(0)  # Non-Accident
        
        print(f"✓ Loaded {len(self.images)} images from {root_dir}")
        print(f"  Accident images: {sum(1 for l in self.labels if l == 1)}")
        print(f"  Non-Accident images: {sum(1 for l in self.labels if l == 0)}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get image and label by index
        
        Returns:
            Tuple of (image_tensor, label)
        """
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a zero tensor on error
            image = torch.zeros(3, *self.target_size)
        
        return image, label


class CSVAccidentDataset(Dataset):
    """
    Dataset for loading images with labels from CSV file
    
    CSV format:
    image_path, accident, severity_score
    path/to/image1.jpg, 0, 0
    path/to/image2.jpg, 1, 75
    path/to/image3.jpg, 1, 45
    ...
    """
    
    def __init__(
        self,
        csv_path: str,
        image_root_dir: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (224, 224)
    ):
        """
        Initialize dataset from CSV file
        
        Args:
            csv_path: Path to CSV file with columns [image_path, accident, severity_score]
            image_root_dir: Optional root directory to prepend to relative image paths
            transform: Torchvision transforms to apply
            target_size: Target image size
        """
        self.csv_path = csv_path
        self.image_root_dir = image_root_dir
        self.target_size = target_size
        
        # Read CSV
        self.df = pd.read_csv(csv_path)
        
        # Validate CSV columns
        required_columns = ['image_path', 'accident', 'severity_score']
        if not all(col in self.df.columns for col in required_columns):
            raise ValueError(f"CSV must have columns: {required_columns}")
        
        # Default transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
        
        print(f"✓ Loaded CSV with {len(self.df)} entries from {csv_path}")
        print(f"  Accident samples: {(self.df['accident'] == 1).sum()}")
        print(f"  Non-Accident samples: {(self.df['accident'] == 0).sum()}")
        print(f"  Severity score range: {self.df['severity_score'].min():.1f}-{self.df['severity_score'].max():.1f}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, float]:
        """
        Get image, accident label, and severity score by index
        
        Returns:
            Tuple of (image_tensor, accident_label, severity_score)
        """
        row = self.df.iloc[idx]
        img_path = row['image_path']
        accident = int(row['accident'])
        severity_score = float(row['severity_score'])
        
        # Prepend root directory if provided
        if self.image_root_dir and not os.path.isabs(img_path):
            img_path = os.path.join(self.image_root_dir, img_path)
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = torch.zeros(3, *self.target_size)
        
        return image, accident, severity_score


def get_data_loaders(
    dataset_root: str,
    dataset_type: str = 'folder',
    csv_path: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    train_split: float = 0.8,
    shuffle_train: bool = True,
    augment: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train and validation data loaders
    
    Args:
        dataset_root: Root directory of dataset
        dataset_type: 'folder' for folder structure or 'csv' for CSV-based
        csv_path: Path to CSV file (required if dataset_type='csv')
        batch_size: Batch size for loaders
        num_workers: Number of workers for data loading
        train_split: Proportion of data for training (default: 0.8)
        shuffle_train: Whether to shuffle training data
        augment: Whether to apply data augmentation
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    
    # Define transforms
    if augment:
        train_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        train_transforms = None
    
    val_transforms = None  # Default transforms used
    
    # Load dataset
    if dataset_type == 'csv':
        if csv_path is None:
            raise ValueError("csv_path required when dataset_type='csv'")
        dataset = CSVAccidentDataset(
            csv_path=csv_path,
            image_root_dir=dataset_root,
            transform=train_transforms,
            target_size=(224, 224)
        )
    else:  # folder structure
        dataset = AccidentImageDataset(
            root_dir=dataset_root,
            transform=train_transforms,
            target_size=(224, 224)
        )
    
    # Split into train and validation
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"\n✓ Data loaders created:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Batch size: {batch_size}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test with folder structure
    print("Testing AccidentImageDataset with folder structure...")
    dataset_root = "e:\\Acident_Detection_Agent\\datasets"
    dataset = AccidentImageDataset(dataset_root)
    print(f"Dataset size: {len(dataset)}")
    
    # Get a sample
    sample_image, label = dataset[0]
    print(f"Sample image shape: {sample_image.shape}, Label: {label}")
    
    # Test data loaders
    print("\nTesting DataLoaders...")
    train_loader, val_loader = get_data_loaders(
        dataset_root=dataset_root,
        dataset_type='folder',
        batch_size=16,
        augment=True
    )
    
    # Get a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels shape: {labels.shape}")

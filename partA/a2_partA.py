import os
import argparse
import numpy as np
import wandb
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset

torch.set_float32_matmul_precision('medium')

# ---------------------------
# Custom Stratified Split Function
# ---------------------------
def custom_stratified_split(dataset, test_size=0.2, random_state=42):
    """
    Splits the dataset indices into training and validation sets in a stratified manner.
    Each class will contribute test_size fraction of its indices to the validation set.
    """
    np.random.seed(random_state)
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    train_idx = []
    val_idx = []
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        np.random.shuffle(cls_indices)
        n_val = int(len(cls_indices) * test_size)
        val_idx.extend(cls_indices[:n_val])
        train_idx.extend(cls_indices[n_val:])
    return np.array(train_idx), np.array(val_idx)

# ---------------------------
# Dynamic Filter Channels Generator
# ---------------------------
def get_conv_channels(filter_org, base_filter):
    """
    Generate a list of filter counts for 5 conv layers based on the filter organization strategy.
    """
    if filter_org == "same":
        return [base_filter] * 5
    elif filter_org == "double":
        return [base_filter * (2 ** i) for i in range(5)]
    elif filter_org == "half":
        return [max(1, base_filter // (2 ** i)) for i in range(5)]
    else:
        raise ValueError("Unknown filter organization: choose from 'same', 'double', or 'half'.")

# ---------------------------
# Define the CNN Model
# ---------------------------
class LitCNN(pl.LightningModule):
    def __init__(self, 
                 in_channels=3,
                 conv_channels=[16, 32, 64, 128, 256],
                 kernel_sizes=3,
                 dense_neurons=128,
                 num_classes=10,
                 conv_activation="ReLU",
                 bn=False,
                 dropout=0.0,
                 image_size=224,
                 data_augment=False):
        super().__init__()
        self.save_hyperparameters()
        
        # Map string names to activation functions
        act_funcs = {
            "ReLU": nn.ReLU,
            "GELU": nn.GELU,
            "SiLU": nn.SiLU,
            "Mish": nn.Mish,
            "LeakyReLU": nn.LeakyReLU
        }
        activation_fn = act_funcs.get(conv_activation, nn.ReLU)
        
        # Ensure kernel_sizes is a list with proper length
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(conv_channels)
        elif isinstance(kernel_sizes, list):
            if len(kernel_sizes) != len(conv_channels):
                raise ValueError("Length of kernel_sizes must equal number of conv layers")
        else:
            raise ValueError("kernel_sizes must be an int or a list of ints")
        
        # Build conv blocks: Conv2d -> [BatchNorm2d] -> Activation -> MaxPool2d
        layers = []
        prev_channels = in_channels
        for out_channels, k in zip(conv_channels, kernel_sizes):
            layers.append(nn.Conv2d(prev_channels, out_channels, kernel_size=k, padding=k//2))
            if bn:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(activation_fn())
            layers.append(nn.MaxPool2d(kernel_size=2))
            # Optionally, you can add dropout here if desired:
            # if dropout > 0.0:
            #     layers.append(nn.Dropout(dropout))
            prev_channels = out_channels
        self.conv_layers = nn.Sequential(*layers)
        
        # Compute flattened dimension after pooling operations
        final_size = image_size // (2 ** len(conv_channels))
        self.flatten_dim = prev_channels * final_size * final_size
        
        # Dense layers
        self.fc1 = nn.Linear(self.flatten_dim, dense_neurons)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.fc2 = nn.Linear(dense_neurons, num_classes)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

# ---------------------------
# Data Loading Function
# ---------------------------
def get_data_loaders(data_dir, image_size, batch_size, data_augment):
    num_workers = min(8, os.cpu_count() - 1)
    prefetch_factor = 2 if num_workers > 0 else None

    # Normalization values for ImageNet (commonly used for natural images)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if data_augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    
    # For validation, only resize and normalize (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    
    # Use custom stratified split so each class is equally represented in validation
    train_idx, val_idx = custom_stratified_split(full_train_dataset, test_size=0.2, random_state=42)
    train_dataset = Subset(full_train_dataset, train_idx)
    # Override transform for validation subset
    val_dataset = Subset(datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=val_transform), val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
    
    return train_loader, val_loader

# ---------------------------
# Main Function with wandb Integration and Dynamic Run Naming
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../inaturalist_12k",
                        help="Path to dataset directory (should contain a 'train' folder with one folder per class)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    # Model hyperparameters (to be swept via wandb)
    parser.add_argument("--conv_activation", type=str, default="ReLU")
    parser.add_argument("--filter_org", type=str, default="double", choices=["same", "double", "half"])
    parser.add_argument("--base_filter", type=int, default=32)
    parser.add_argument("--kernel_sizes", type=int, default=3)
    parser.add_argument("--dense_neurons", type=int, default=128)
    parser.add_argument("--bn", type=lambda x: x.lower() == "true", default=False,
                        help="Enable Batch Normalization")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--data_augment", type=lambda x: x.lower() == "true", default=False,
                        help="Enable data augmentation")
    parser.add_argument("--max_epochs", type=int, default=50)
    args = parser.parse_args()
    
    # Initialize wandb (ensure your project name is "assignment_2")
    wandb.init(project="assignment_2", config=vars(args), settings=wandb.Settings(start_method="thread"))
    config = wandb.config

    # Set a dynamic run name based on key hyperparameters
    dynamic_run_name = (
        f"act_{config.conv_activation}_"
        f"org_{config.filter_org}_"
        f"bf_{config.base_filter}_"
        f"bn_{config.bn}_"
        f"dr_{config.dropout}_"
        f"aug_{config.data_augment}_"
        f"bs_{config.batch_size}"
    )
    wandb.run.name = dynamic_run_name

    # Dynamically generate conv_channels based on the filter organization strategy
    conv_channels = get_conv_channels(config.filter_org, config.base_filter)
    
    # Get train and validation data loaders (using the 'train' folder only)
    train_loader, val_loader = get_data_loaders(config.data_dir, config.image_size, config.batch_size, config.data_augment)
    
    # Initialize the model using hyperparameters from wandb config
    model = LitCNN(
        in_channels=3,
        conv_channels=conv_channels,
        kernel_sizes=config.kernel_sizes,
        dense_neurons=config.dense_neurons,
        num_classes=10,
        conv_activation=config.conv_activation,
        bn=config.bn,
        dropout=config.dropout,
        image_size=config.image_size,
        data_augment=config.data_augment
    )
    
    # Optionally, disable torch.compile if you suspect any issues.
    # model = torch.compile(model)

    # Set up the PyTorch Lightning Trainer with wandb logging
    wandb_logger = pl.loggers.WandbLogger(project="assignment_2")
    trainer = pl.Trainer(
        precision=16,
        check_val_every_n_epoch=5,
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
        log_every_n_steps=5
    )
    
    # Start training; wandb will log metrics and generate the required plots
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()

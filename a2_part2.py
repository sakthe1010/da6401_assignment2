import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import wandb

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
# Data Loading Function
# ---------------------------
def get_data_loaders(data_dir, image_size, batch_size, data_augment):
    num_workers = min(8, os.cpu_count() - 1)
    prefetch_factor = 2 if num_workers > 0 else None

    # Normalization values for ImageNet
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
    
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    full_train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    
    # Use custom stratified split so each class is equally represented in validation
    train_idx, val_idx = custom_stratified_split(full_train_dataset, test_size=0.2, random_state=42)
    train_dataset = Subset(full_train_dataset, train_idx)
    # For validation, override the transform to avoid augmentation
    val_dataset = Subset(datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=val_transform), val_idx)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True, persistent_workers=True, prefetch_factor=prefetch_factor)
    
    return train_loader, val_loader

# ---------------------------
# PyTorch Lightning Module for Fine-Tuning Pre-Trained ResNet50
# ---------------------------
class LitFinetuneResNet(pl.LightningModule):
    def __init__(self, freeze_strategy="strat2", num_classes=10, lr=1e-3):
        """
        freeze_strategy options:
          - "strat1": Freeze all layers except the final fully connected layer.
          - "strat2": Freeze all layers except layer4 and the final fc layer.
          - "full": Fine-tune the entire network.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        
        # Load a pre-trained ResNet50 model
        self.model = torchvision.models.resnet50(pretrained=True)
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)
        
        # Apply freeze strategy (default "strat2")
        self.apply_freeze_strategy(freeze_strategy)
    
    def apply_freeze_strategy(self, strategy):
        if strategy == "strat1":
            # Freeze all parameters except the final fc layer.
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif strategy == "strat2":  # Freeze all except layer4 and fc
            for name, param in self.model.named_parameters():
                if not (name.startswith("layer4") or name.startswith("fc")):
                    param.requires_grad = False
        elif strategy == "full":
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            raise ValueError(f"Unknown freeze strategy: {strategy}")
    
    def forward(self, x):
        return self.model(x)
    
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
        # Only update parameters with requires_grad=True.
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
        return optimizer

# ---------------------------
# Main Function with wandb Integration and Dynamic Run Naming
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./inaturalist_12k",
                        help="Path to dataset directory (should contain a 'train' folder with one folder per class)")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    # Using a short freeze strategy name ("strat2")
    parser.add_argument("--freeze_strategy", type=str, default="strat2",
                        choices=["strat1", "strat2", "full"],
                        help="Freeze strategy to use for fine-tuning")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data_augment", type=lambda x: x.lower() == "true", default=False,
                        help="Enable data augmentation")
    parser.add_argument("--max_epochs", type=int, default=20)
    args = parser.parse_args()
    
    # Initialize wandb (project name "assignment_2")
    wandb.init(project="assignment_2", config=vars(args), settings=wandb.Settings(start_method="thread"))
    config = wandb.config

    # Create a dynamic run name without including the freeze strategy (since it's constant)
    dynamic_run_name = f"lr_{config.lr}_aug_{config.data_augment}_bs_{config.batch_size}"
    wandb.run.name = dynamic_run_name

    # Get train and validation data loaders (using the 'train' folder)
    train_loader, val_loader = get_data_loaders(config.data_dir, config.image_size, config.batch_size, config.data_augment)
    
    # Initialize the fine-tuning model using hyperparameters from wandb config
    model = LitFinetuneResNet(
        freeze_strategy=config.freeze_strategy,
        num_classes=10,
        lr=config.lr
    )
    
    # Set up the PyTorch Lightning Trainer with wandb logging
    wandb_logger = pl.loggers.WandbLogger(project="assignment_2")
    trainer = pl.Trainer(
        precision=16,
        check_val_every_n_epoch=2,
        max_epochs=config.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
        log_every_n_steps=5
    )
    
    # Start training; wandb will log metrics and generate plots.
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()

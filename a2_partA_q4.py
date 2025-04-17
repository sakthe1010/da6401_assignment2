import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
import wandb  # optional if you want logging
import math

###############################################################################
# 1) LitCNN CLASS (same as in a2_partA.py but simplified to your best config)
###############################################################################
class LitCNN(pl.LightningModule):
    def __init__(
        self,
        in_channels=3,
        conv_channels=[32, 32, 32, 32, 32],  # 'same' => 5 conv layers each with 32 filters
        kernel_size=3,
        dense_neurons=256,
        num_classes=10,
        activation="Mish",
        bn=True,
        dropout=0.3,
        image_size=224,
    ):
        super().__init__()
        self.save_hyperparameters()

        # 1. Activation Setup
        act_dict = {
            "ReLU": torch.nn.ReLU,
            "GELU": torch.nn.GELU,
            "SiLU": torch.nn.SiLU,
            "Mish": torch.nn.Mish,
            "LeakyReLU": torch.nn.LeakyReLU,
        }
        activation_fn = act_dict.get(activation, torch.nn.ReLU)

        # 2. Build Convolution Layers
        layers = []
        prev_channels = in_channels
        for out_ch in conv_channels:
            layers.append(torch.nn.Conv2d(prev_channels, out_ch, kernel_size=kernel_size, padding=kernel_size // 2))
            if bn:
                layers.append(torch.nn.BatchNorm2d(out_ch))
            layers.append(activation_fn())
            layers.append(torch.nn.MaxPool2d(kernel_size=2))
            prev_channels = out_ch
        self.conv_layers = torch.nn.Sequential(*layers)

        # 3. Compute flatten dimension
        #    After 5 max pools on a 224x224 image, final spatial size = 224/(2^5) = 7
        final_size = image_size // (2**len(conv_channels))
        self.flatten_dim = prev_channels * final_size * final_size

        # 4. Dense + Output
        self.fc1 = torch.nn.Linear(self.flatten_dim, dense_neurons)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()
        self.fc2 = torch.nn.Linear(dense_neurons, num_classes)

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
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


###############################################################################
# 2) DATASET + DATALOADERS (stratified split for train/val)
###############################################################################
def stratified_split(dataset, test_size=0.2, random_state=42):
    np.random.seed(random_state)
    targets = np.array([s[1] for s in dataset.samples])  # dataset.targets for new Torch versions
    classes = np.unique(targets)
    train_idx, val_idx = [], []
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        np.random.shuffle(cls_indices)
        n_val = int(len(cls_indices) * test_size)
        val_idx.extend(cls_indices[:n_val])
        train_idx.extend(cls_indices[n_val:])
    return train_idx, val_idx

def get_data_loaders(data_dir, image_size=224, batch_size=64):
    # Normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # No data augmentation, per your best config
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Full dataset from train folder
    full_dataset = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)

    # Stratified split
    train_idx, val_idx = stratified_split(full_dataset, test_size=0.2, random_state=42)
    train_ds = torch.utils.data.Subset(full_dataset, train_idx)
    val_ds = torch.utils.data.Subset(full_dataset, val_idx)
    # Override transform for val subset
    # (only needed if you want a separate transform, but we have the same here)
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Test set
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader


###############################################################################
# 3) MAIN SCRIPT: Train (from scratch) -> Evaluate -> Plot Grid
###############################################################################
def train_and_evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./inaturalist_12k",
                        help="Path to dataset folder with 'train'/'test'.")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=10, 
                        help="Number of epochs to train. Increase if needed.")
    parser.add_argument("--save_grid", type=str, default="prediction_grid.png")
    args = parser.parse_args()

    # 1. Get Data
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=args.data_dir,
        image_size=224,
        batch_size=args.batch_size
    )

    # 2. Build Model with best config
    model = LitCNN(
        in_channels=3,
        conv_channels=[32, 32, 32, 32, 32],
        kernel_size=3,
        dense_neurons=256,
        num_classes=10,
        activation="Mish",
        bn=True,
        dropout=0.3,
        image_size=224,
    )

    # 3. Train Model
    wandb.init(project="assignment_2", name="best_config_run_from_scratch", reinit=True)
    wandb_logger = pl.loggers.WandbLogger(project="assignment_2")

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        logger=wandb_logger,
        log_every_n_steps=10
    )
    trainer.fit(model, train_loader, val_loader)

    # 4. Evaluate Model on Test Set
    test_acc = evaluate_on_test(model, test_loader)
    print(f"Test Accuracy with best config (trained from scratch): {test_acc:.2f}%")

    # 5. Plot 10x3 Grid
    plot_predictions_grid(model, test_loader, args.save_grid)

    wandb.finish()


def evaluate_on_test(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return (correct / total) * 100.0


def plot_predictions_grid(model, test_loader, save_path="prediction_grid.png", num_samples=30):
    """
    Saves a 10x3 grid of images from the test set with true vs predicted labels.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Grab one batch
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    # In case the batch is smaller, clamp to `num_samples`
    images = images[:num_samples].to(device)
    labels = labels[:num_samples].to(device)

    with torch.no_grad():
        logits = model(images)
        preds = logits.argmax(dim=1)

    class_names = test_loader.dataset.classes

    # Normalization (inverse) for display
    inv_normalize = transforms.Normalize(
        mean=[-m / s for (m, s) in zip([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])],
        std=[1 / s for s in [0.229, 0.224, 0.225]]
    )

    # 10 rows x 3 cols if we want 30 images
    rows = 10
    cols = 3
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(cols * 3, rows * 3))

    for idx, ax in enumerate(axes.flat):
        if idx >= num_samples:
            break
        img = images[idx].detach().cpu()
        img = inv_normalize(img)
        npimg = img.numpy().transpose((1, 2, 0))
        npimg = np.clip(npimg, 0, 1)

        true_label = class_names[labels[idx].item()]
        pred_label = class_names[preds[idx].item()]

        ax.imshow(npimg)
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    print(f"Saved prediction grid to: {save_path}")


if __name__ == "__main__":
    train_and_evaluate()

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

class LitCNN(pl.LightningModule):
    def __init__(self, 
                 in_channels=3,
                 conv_channels=[32, 64, 128, 256, 512],
                 kernel_sizes=3,            # Can be an int or list of ints
                 dense_neurons=128,
                 num_classes=10,
                 conv_activation=nn.ReLU,   # Activation for conv layers
                 dense_activation=None,     # Optional activation for dense layer (e.g., nn.ReLU)
                 image_size=224):           # Expected input image size (e.g., 224x224)
        super().__init__()
        self.save_hyperparameters()  # Save all init parameters for logging
        
        self.conv_layers = nn.ModuleList()
        prev_channels = in_channels

        # Ensure kernel_sizes is a list matching the number of conv layers.
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * len(conv_channels)
        elif isinstance(kernel_sizes, list):
            if len(kernel_sizes) != len(conv_channels):
                raise ValueError("Length of kernel_sizes must equal number of conv layers")
        else:
            raise ValueError("kernel_sizes must be an int or a list of ints")
        
        # Create 5 convolution blocks
        for out_channels, k in zip(conv_channels, kernel_sizes):
            block = nn.Sequential(
                nn.Conv2d(prev_channels, out_channels, kernel_size=k, padding=k // 2),
                conv_activation(),
                nn.MaxPool2d(kernel_size=2)  # Reduces spatial dimensions by 2
            )
            self.conv_layers.append(block)
            prev_channels = out_channels

        # Compute final spatial dimensions after 5 max-pooling operations.
        final_size = image_size // (2 ** len(conv_channels))
        self.flatten_dim = prev_channels * final_size * final_size

        # Dense layers: one hidden dense layer followed by the output layer.
        self.fc1 = nn.Linear(self.flatten_dim, dense_neurons)
        self.fc2 = nn.Linear(dense_neurons, num_classes)
        self.dense_activation = dense_activation

    def forward(self, x):
        # Pass through convolution blocks
        for block in self.conv_layers:
            x = block(x)
        # Flatten the output for dense layers
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        if self.dense_activation is not None:
            x = self.dense_activation(x)
        x = self.fc2(x)
        return x

    # Training step: compute loss and log it
    def training_step(self, batch, batch_idx):
        print("→ Training on device:", self.device)
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    # Validation step: compute loss and log it
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("val_loss", loss)
        return loss

    # Configure the optimizer
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
    
if __name__ == "__main__":
    # Check GPU availability
    if torch.cuda.is_available():
        print("GPU is available!")
        print("Device Name:", torch.cuda.get_device_name(0))
    else:
        print("GPU is not available.")

    # Instantiate the model
    model = LitCNN(
        in_channels=3,
        conv_channels=[32, 64, 128, 256, 512],
        kernel_sizes=3,
        dense_neurons=128,
        num_classes=10,
        conv_activation=nn.ReLU,
        dense_activation=None,
        image_size=224
    )
    print(model)
    print("Model parameters are on device:", next(model.parameters()).device)

    # Dummy training data
    x = torch.randn(100, 3, 224, 224)
    y = torch.randint(0, 10, (100,))
    train_loader = DataLoader(TensorDataset(x, y), batch_size=8, num_workers=27)

    # Train using Lightning Trainer
    trainer = pl.Trainer(accelerator="gpu", devices=1, max_epochs=10, precision="16-mixed", log_every_n_steps=1)
    trainer.fit(model, train_loader)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
import time

# Read data
data_path = "C:\\Users\\sagyi\\Downloads\\data_all_params.txt"
data = pd.read_csv(data_path)

# Split features and targets
x = data.iloc[:, 5:]  # input features
y = data.iloc[:, :5]  # targets

# Split data into 80% training, 10% validation, and 10% testing
X_train, X_temp, y_train, y_temp = train_test_split(x, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# Convert to tensors
X_train_scaled = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
X_val_scaled = torch.tensor(X_val_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)
y_val = torch.tensor(y_val.values, dtype=torch.float32)

# Dataset class
class DetectorDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Create datasets and dataloaders
train_dataset = DetectorDataset(X_train_scaled, y_train)
test_dataset = DetectorDataset(X_test_scaled, y_test)
val_dataset = DetectorDataset(X_val_scaled, y_val)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, persistent_workers=True)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, persistent_workers=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

# Define the neural network model
class RegressionModel(pl.LightningModule):
    def __init__(self, num_features, num_outputs, learning_rate=0.001):
        super().__init__()
        
        self.fc1 = nn.Linear(num_features, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_outputs) 
        self.learning_rate = learning_rate
        self.save_hyperparameters()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        # Log training loss per batch
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = nn.functional.mse_loss(y_hat, y)
        # Log validation loss
        self.log('val_loss', val_loss, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = nn.functional.mse_loss(y_hat, y)
        # Log test loss
        self.log('test_loss', test_loss, on_epoch=True, on_step=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# Main execution block to handle multiprocessing issues on Windows
if __name__ == '__main__':

    start_time = time.time()
    # TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="regression_model")

    # Checkpoint callback to save best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='my_checkpoints',
        filename='best_model',
        save_top_k=1,
        mode='min'
    )

    model = RegressionModel(num_features=X_train_scaled.shape[1], num_outputs=y_train.shape[1])
    
    trainer = pl.Trainer(
        max_epochs=100,
        logger=logger,
        callbacks=[checkpoint_callback],
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    end_time = time.time()
    print(f"Training completed in {(end_time - start_time)/60:.2f} minutes")


    # Test the model on the test dataset
    trainer.test(model, dataloaders=test_dataloader)





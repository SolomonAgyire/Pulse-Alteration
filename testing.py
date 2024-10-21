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
from torch.utils.data import Dataset, DataLoader


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
X_train_scaled = scaler.fit_transform(X_train)  # Used for loading the model
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)

# Convert to tensors
X_test_scaled = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

# Dataset class
class DetectorDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

# Create the test dataloader
test_dataset = DetectorDataset(X_test_scaled, y_test)
test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=2, persistent_workers=True)

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

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = nn.functional.mse_loss(y_hat, y)
        # Log test loss
        self.log('test_loss', test_loss, on_epoch=True, on_step=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


# Properly load the trained model from the checkpoint
if __name__ == '__main__':
    # TensorBoard logger
    logger = TensorBoardLogger("tb_logs", name="regression_model_testing")
    
    # Load the best model checkpoint
    best_model_path = 'my_checkpoints/best_model.ckpt'
    model = RegressionModel.load_from_checkpoint(best_model_path,num_features=X_train_scaled.shape[1],num_outputs=y_train.shape[1])
                                                  
                                                                  
    # Test the model on the test dataset with logger
    trainer = pl.Trainer(logger=logger)
    for epoch in range(3):
        print(f"Testing: Epoch {epoch + 1}")
        trainer.test(model, dataloaders=test_dataloader)

         # Log the test loss with the epoch number in the tag
        test_loss = trainer.callback_metrics['test_loss'].item()
        trainer.logger.experiment.add_scalar(f"test_loss_epoch_{epoch + 1}", test_loss, epoch + 1)

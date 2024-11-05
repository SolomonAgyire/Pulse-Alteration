# Libraries
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
import matplotlib.pyplot as plt
from PIL import Image
import io



# Load  + normalize 
file_path = r"C:\Users\sagyi\Downloads\x1.txt"
data = pd.read_csv(file_path, sep=' ', header=None)

data_path = r"C:\Users\sagyi\Downloads\data_all_params.txt"
est_data = pd.read_csv(data_path, sep=',', header=0).iloc[:, :5]

data_norm = data.div(data.max(axis=1), axis=0)

# Split data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(data_norm, est_data, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Converting to PyTorch tensors
X_train_t = torch.tensor(X_train.values, dtype=torch.float32)
X_val_t = torch.tensor(X_val.values, dtype=torch.float32)
X_test_t = torch.tensor(X_test.values, dtype=torch.float32)
y_train_t = torch.tensor(y_train.values, dtype=torch.float32)
y_val_t = torch.tensor(y_val.values, dtype=torch.float32)
y_test_t = torch.tensor(y_test.values, dtype=torch.float32)

# Define dataset and dataloader 
class DetectorDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

#Dataset
train_dataset = DetectorDataset(X_train_t, y_train_t)
val_dataset = DetectorDataset(X_val_t, y_val_t)
test_dataset = DetectorDataset(X_test_t, y_test_t)

#Dataloader
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, persistent_workers = True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, persistent_workers = True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, persistent_workers = True)

# Define the neural network model
class PulseParameterNN(pl.LightningModule):
    def __init__(self):
        super(PulseParameterNN, self).__init__()
        self.fc1 = nn.Linear(250, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 5)
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = self.loss_fn(y_hat, y)
        self.log('val_loss', val_loss, on_epoch=True)
        return val_loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.loss_fn(y_hat, y)
        self.log('test_loss', test_loss, on_step=True, on_epoch=True)
        return {'test_loss': test_loss, 'predictions': y_hat, 'features': x}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

# Function (Formula) for pulse reconstruction
def reconstruct_pulse(t, params):
    """Reconstruct the pulse based on the model's predicted parameters."""
    A1, k1, k2, T1, O = params
    return A1 * (np.exp(-k1 * (t - T1)) / (1 + np.exp(-k2 * (t - T1)))) + O


def main():
    
    #log onto tensor board
    logger = TensorBoardLogger("tb_logs", name="my_model")
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='my_checkpoint',
        filename='best_model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        mode='min'
    )

    
    trainer = Trainer(
        max_epochs=100,
        logger=logger,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True
    )

    # Model training
    model = PulseParameterNN()
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader, ckpt_path=checkpoint_callback.best_model_path)

    
    
    # loading the best model for pulse reconstruction
    best_model_path = "my_checkpoint/best_model-epoch=36-val_loss=0.00.ckpt"
    best_model = PulseParameterNN.load_from_checkpoint(best_model_path)
    best_model.eval()
    
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2, persistent_workers=True)
    
    # Time vector
    t = np.linspace(0, 1, 250)
    
    mse_scores = []

    with torch.no_grad():
        for i, (x, y_true_params) in enumerate(test_dataloader):
            y_pred_params = best_model(x).cpu().numpy()
            reconstructed_pulses = [reconstruct_pulse(t, params) for params in y_pred_params]
            actual_pulses = x.cpu().numpy()

            for j, (actual, reconstructed) in enumerate(zip(actual_pulses, reconstructed_pulses)):
                mse = root_mean_squared_error(actual, reconstructed)
                mse_scores.append(mse)

    # Check the MSE score
    print(f"Average MSE for reconstructed pulses: {np.mean(mse_scores)}")

if __name__ == '__main__':
    main()

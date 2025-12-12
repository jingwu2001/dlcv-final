import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from data_loader import DistanceDataset
from model import DistanceRegressor

# Argument parsing if needed, or simple implementation
USE_DEPTH = True
BACKBONE = 'convnext_tiny' # resnet50, convnext_tiny, etc.

dataset = DistanceDataset('../data/train', '../data/train/train_dist_est.json', rgb=True, depth=USE_DEPTH)
loader = DataLoader(dataset, batch_size=36, shuffle=True, num_workers=4)

input_channels = 5 + int(USE_DEPTH) # 5 (RGB + 2 Masks) + Depth
print(f"Initializing model with backbone={BACKBONE}, input_channels={input_channels}")

model = DistanceRegressor(input_channels=input_channels, backbone=BACKBONE, pretrained=True).cuda()
# Commenting out load_state_dict for new training since architecture/channels changed
# model.load_state_dict(torch.load('ckpt/epoch_5_iter_6831.pth')) 
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

print(f"Dataset size: {len(dataset)}")
print(f"Number of batches: {len(loader)}")

# Training loop
num_epochs = 1
# Limit batches for dry run if needed, but for now we'll just run it.
for epoch in range(num_epochs):
    print(f"Starting epoch {epoch + 1}/{num_epochs}")
    model.train()
    total_loss = 0
    epoch_loss = 0
    num_batches = len(loader)
    log_interval = 50
    save_interval = 3000

    for i, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.cuda(), targets.cuda()
        preds = model(inputs)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        epoch_loss += batch_loss

        if (i + 1) % log_interval == 0 or (i + 1) == num_batches:
            print(f"[Epoch {epoch+1}/{num_epochs}] "
                  f"Iter {i+1}/{num_batches} "
                  f"Loss: {batch_loss:.4f}")

        if (i + 1) % save_interval == 0 and epoch > 3:
            torch.save(model.state_dict(), f"ckpt/epoch_{epoch+1}_iter_{i+1}.pth")
            print(f"Model saved at epoch {epoch+1}, iteration {i+1}")

    avg_loss = epoch_loss / num_batches
    print(f"[Epoch {epoch+1}/{num_epochs}] Average Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), f"ckpt/epoch_{epoch+1}_iter_{i+1}.pth")
    print(f"Model saved at epoch {epoch+1}, iteration {i+1}")
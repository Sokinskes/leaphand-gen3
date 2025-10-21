import torch
import torch.nn as nn
import numpy as np
from bc_planner import BCPlanner
import os
import time
import json
import matplotlib.pyplot as plt
from glob import glob

def batch_loader(traj_arr, cond_arr, batch_size=32, shuffle=True):
    N = traj_arr.shape[0]
    idxs = np.arange(N)
    if shuffle:
        np.random.shuffle(idxs)
    for i in range(0, N, batch_size):
        batch_idx = idxs[i:i+batch_size]
        yield {
            'traj': traj_arr[batch_idx],  # [B, action_dim]
            'cond': cond_arr[batch_idx]   # [B, cond_dim]
        }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load enhanced aggregated data from data.npz
print("Loading enhanced data from data.npz...")
data = np.load('data.npz')
trajectories = data['trajectories']
poses = data['poses']
pcs = data['pcs']
tactiles = data['tactiles']
conds = np.concatenate([poses, pcs, tactiles], axis=1)  # [N, 3+6144+100]

print(f"Loaded enhanced data: trajectories {trajectories.shape}, conds {conds.shape}")

# Normalize condition vectors
cond_mean = conds.mean(axis=0, keepdims=True)
cond_std = conds.std(axis=0, keepdims=True) + 1e-6
conds = (conds - cond_mean) / cond_std
np.savez('cond_stats.npz', mean=cond_mean, std=cond_std)

# Random split for enhanced data (since video-level meta is not directly applicable)
N = trajectories.shape[0]
val_split = 0.1
perm = np.random.permutation(N)
n_val = int(N * val_split)
val_idx = perm[:n_val]
train_idx = perm[n_val:]
train_traj = trajectories[train_idx]
train_conds = conds[train_idx]
val_traj = trajectories[val_idx]
val_conds = conds[val_idx]
print(f"Random split: training {len(train_idx)}, validation {len(val_idx)}")

# Infer dimensions
action_dim = trajectories.shape[1]
cond_dim = conds.shape[1]

print(f"Inferred action_dim={action_dim}, cond_dim={cond_dim}")

# Create model
model = BCPlanner(cond_dim=cond_dim, action_dim=action_dim).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

num_epochs = 1000
batch_size = 32
best_val = float('inf')
patience = 20
stale = 0
run_name = time.strftime('run_bc_%Y%m%d_%H%M%S')
run_dir = os.path.join('runs', run_name)
os.makedirs(run_dir, exist_ok=True)
log_path = os.path.join(run_dir, 'log.json')
history = {'train_loss': [], 'val_loss': []}

for epoch in range(num_epochs):
    total_loss = 0
    n_batch = 0
    for batch in batch_loader(train_traj, train_conds, batch_size=batch_size):
        traj = torch.tensor(batch['traj'], dtype=torch.float32, device=device)
        cond = torch.tensor(batch['cond'], dtype=torch.float32, device=device)
        pred = model(cond)
        loss = nn.MSELoss()(pred, traj)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * traj.shape[0]
        n_batch += traj.shape[0]
    train_loss = total_loss / n_batch

    model.eval()
    val_loss = 0
    val_n = 0
    with torch.no_grad():
        for batch in batch_loader(val_traj, val_conds, batch_size=batch_size, shuffle=False):
            traj = torch.tensor(batch['traj'], dtype=torch.float32, device=device)
            cond = torch.tensor(batch['cond'], dtype=torch.float32, device=device)
            pred = model(cond)
            loss = nn.MSELoss()(pred, traj)
            val_loss += loss.item() * traj.shape[0]
            val_n += traj.shape[0]
    val_loss = val_loss / max(1, val_n)
    print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

    if val_loss < best_val:
        best_val = val_loss
        stale = 0
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(run_dir, 'best_model.pth'))
        print(f"Saved best_model.pth at epoch {epoch}, val_loss={val_loss:.6f}")
    else:
        stale += 1

    if stale >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    with open(log_path, 'w') as f:
        json.dump(history, f)
    plt.figure()
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.legend()
    plt.savefig(os.path.join(run_dir, 'loss.png'))
    plt.close()
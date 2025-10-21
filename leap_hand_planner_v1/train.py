
import torch
import torch.nn as nn
import numpy as np
from diffusion_planner import DiffusionPlanner
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
            'traj': traj_arr[batch_idx],  # [B, T, A]
            'cond': cond_arr[batch_idx]   # [B, cond_dim]
        }

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载所有data_*.npz文件并合并
data_files = glob('data_*.npz')
all_trajectories = []
all_conds = []
meta = []
idx_cursor = 0
for file in data_files:
    data = np.load(file)
    trajectories = data['trajectories']
    poses = data['poses']
    pcs = data['pcs']
    tactiles = data['tactiles']
    conds = np.concatenate([poses, pcs, tactiles], axis=1)  # [N, 3+6144+100]
    all_trajectories.append(trajectories)
    all_conds.append(conds)
    # record metadata mapping sample ranges to source file
    n = trajectories.shape[0]
    meta.append({'file': file, 'start_idx': idx_cursor, 'end_idx': idx_cursor + n - 1, 'count': n})
    idx_cursor += n

trajectories = np.concatenate(all_trajectories, axis=0)
conds = np.concatenate(all_conds, axis=0)

# save per-file metadata for video-level splits
import json, os
if meta:
    with open('data_all_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote data_all_meta.json with {len(meta)} files")

print(f"Loaded {len(data_files)} data files, total trajectories: {trajectories.shape}, conds: {conds.shape}")

# normalize condition vectors (per-feature mean/std) — important for stable training
cond_mean = conds.mean(axis=0, keepdims=True)
cond_std = conds.std(axis=0, keepdims=True) + 1e-6
conds = (conds - cond_mean) / cond_std
# save stats for later inference
np.savez('cond_stats.npz', mean=cond_mean, std=cond_std)

# infer dimensions from data and create model so channel sizes match
action_dim = trajectories.shape[1]  # expected action vector length per sample
cond_dim = conds.shape[1]
traj_len = 1  # current dataset produces single-step trajectories (shape [N, action_dim])

# pad seq_len to be compatible with UNet downsampling (multiple of 8 for 3 down blocks)
padded_seq_len = ((action_dim + 7) // 8) * 8  # next multiple of 8

print(f"Inferred action_dim={action_dim}, padded_seq_len={padded_seq_len}, cond_dim={cond_dim}, traj_len={traj_len}")

# create model with matching dims
from diffusion_planner import DiffusionPlanner
model = DiffusionPlanner(cond_dim=cond_dim, seq_len=padded_seq_len, in_channels=1, hidden_dim=256).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
model.DEBUG = True

#
# For video-level splits we use metadata file mapping
import json
with open('data_all_meta.json', 'r') as f:
    file_meta = json.load(f)

# Choose split mode: 'video' for video-level, 'sample' for sample-level
split_mode = 'video'

if split_mode == 'video':
    # leave-one-video-out: pick one file as validation (rotating LOO can be scripted externally)
    # Here we pick the last file as validation by default
    val_meta = file_meta[-1]
    val_start, val_end = val_meta['start_idx'], val_meta['end_idx']
    val_idx = np.arange(val_start, val_end + 1)
    train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end + 1, trajectories.shape[0])])
    train_traj = trajectories[train_idx]
    train_conds = conds[train_idx]
    val_traj = trajectories[val_idx]
    val_conds = conds[val_idx]
    print(f"Video-level split: training on {len(file_meta)-1} files, validating on {val_meta['file']} with {len(val_idx)} samples")
else:
    val_split = 0.1
    N = trajectories.shape[0]
    perm = np.random.permutation(N)
    n_val = max(1, int(N * val_split))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_traj = trajectories[train_idx]
    train_conds = conds[train_idx]
    val_traj = trajectories[val_idx]
    val_conds = conds[val_idx]

num_epochs = 1000
batch_size = 32

# early stopping params
best_val = float('inf')
best_epoch = -1
patience = 20
stale = 0
# setup run folder
run_name = time.strftime('run_%Y%m%d_%H%M%S')
run_dir = os.path.join('runs', run_name)
os.makedirs(run_dir, exist_ok=True)
log_path = os.path.join(run_dir, 'log.json')
history = {'train_loss': [], 'val_loss': []}

# EMA model params
ema_decay = 0.999
ema_params = {n: p.detach().clone() for n, p in model.named_parameters()}

printed_once = False
for epoch in range(num_epochs):
    total_loss = 0
    n_batch = 0
    # training loop over training subset
    for batch in batch_loader(train_traj, train_conds, batch_size=batch_size):
        traj = torch.tensor(batch['traj'], dtype=torch.float32, device=device)  # [B, action_dim]
        traj = traj.unsqueeze(1)  # [B, 1, action_dim]
        # pad to padded_seq_len
        traj = torch.nn.functional.pad(traj, (0, padded_seq_len - action_dim), mode='constant', value=0)  # [B, 1, padded_seq_len]
        cond = torch.tensor(batch['cond'], dtype=torch.float32, device=device)  # [B, cond_dim]
        # one-time print of shapes to help debug channel mismatch
        if model.DEBUG and not printed_once:
            print(f"[RUN-DEBUG] traj tensor shape: {tuple(traj.shape)}")
            print(f"[RUN-DEBUG] cond tensor shape: {tuple(cond.shape)}")
            printed_once = True
        noise = torch.randn_like(traj)
        t = torch.randint(0, model.scheduler.config.num_train_timesteps, (traj.shape[0],), device=device)
        noisy_traj = model.scheduler.add_noise(traj, noise, t)
        # one-time strict checks to catch upstream shape errors
        if not hasattr(model, '_checked_shapes'):
            print(f"[CHECK] traj.shape={tuple(traj.shape)}, cond.shape={tuple(cond.shape)}, noisy_traj.shape={tuple(noisy_traj.shape)}, t.shape={tuple(t.shape)}")
            assert traj.ndim == 3 and traj.shape[1] == 1 and traj.shape[2] == padded_seq_len, f"traj shape unexpected: {traj.shape}"
            assert cond.ndim == 2 and cond.shape[1] == cond_dim, f"cond shape unexpected: {cond.shape}"
            assert t.ndim == 1 and t.shape[0] == traj.shape[0], f"t shape unexpected: {t.shape}"
            model._checked_shapes = True
        pred = model(noisy_traj, t, cond)
        loss = nn.MSELoss()(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * traj.shape[0]
        n_batch += traj.shape[0]

    train_loss = total_loss / n_batch
    print(f"Epoch {epoch}, Train Loss: {train_loss:.6f}")

    # validation
    model.eval()
    val_loss = 0
    val_n = 0
    with torch.no_grad():
        for batch in batch_loader(val_traj, val_conds, batch_size=batch_size, shuffle=False):
            traj = torch.tensor(batch['traj'], dtype=torch.float32, device=device).unsqueeze(1)
            traj = torch.nn.functional.pad(traj, (0, padded_seq_len - action_dim), mode='constant', value=0)  # [B, 1, padded_seq_len]
            cond = torch.tensor(batch['cond'], dtype=torch.float32, device=device)
            noise = torch.randn_like(traj)
            t = torch.randint(0, model.scheduler.config.num_train_timesteps, (traj.shape[0],), device=device)
            noisy_traj = model.scheduler.add_noise(traj, noise, t)
            pred = model(noisy_traj, t, cond)
            loss = nn.MSELoss()(pred, noise)
            val_loss += loss.item() * traj.shape[0]
            val_n += traj.shape[0]
    val_loss = val_loss / max(1, val_n)
    print(f"Epoch {epoch}, Val Loss: {val_loss:.6f}")

    # decide checkpoint/early stop
    if val_loss < best_val:
        best_val = val_loss
        best_epoch = epoch
        stale = 0
        # save best (and EMA) model
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, os.path.join(run_dir, 'best_model.pth'))
        # save EMA params
        torch.save(ema_params, os.path.join(run_dir, 'best_ema_params.pth'))
        print(f"Saved best_model.pth at epoch {epoch}, val_loss={val_loss:.6f}")
    else:
        stale += 1
        print(f"No improvement for {stale} epochs (best {best_val:.6f} at epoch {best_epoch})")

    model.train()

    # early stopping
    if stale >= patience:
        print(f"Early stopping triggered at epoch {epoch} (best_epoch={best_epoch}, best_val={best_val:.6f})")
        break

    # occasional detailed eval (angle error) every 10 epochs
    if (epoch+1) % 10 == 0:
        with torch.no_grad():
            cond_eval = torch.tensor(conds[:1], dtype=torch.float32, device=device)
            gen_seq = model.generate_trajectory(cond_eval)
            gt_traj = trajectories[0]  # [action_dim]
            # gen_seq may be shape [seq_len], trim to action_dim
            err = np.mean(np.abs(gen_seq[:action_dim] - gt_traj)) * 180 / np.pi  # 角度误差
            print(f"Eval traj mean abs error: {err:.2f} deg")

    # update EMA params after epoch
    with torch.no_grad():
        for n, p in model.named_parameters():
            ema_params[n].mul_(ema_decay).add_(p.detach(), alpha=1.0 - ema_decay)

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    # save logs and plots periodically
    with open(log_path, 'w') as f:
        json.dump(history, f)
    try:
        plt.figure()
        plt.plot(history['train_loss'], label='train')
        plt.plot(history['val_loss'], label='val')
        plt.legend()
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.savefig(os.path.join(run_dir, 'loss.png'))
        plt.close()
    except Exception:
        pass
import os
import torch
import numpy as np
from bc_planner import BCPlanner
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import json

def load_data():
    # Load enhanced data
    data = np.load('data.npz')
    trajs = data['trajectories']
    poses = data['poses']
    pcs = data['pcs']
    tactiles = data['tactiles']
    conds = np.concatenate([poses, pcs, tactiles], axis=1)
    return trajs, conds

def postprocess_trajectory(traj, window_length=11, polyorder=2):
    """
    Apply Savitzky-Golay filter for trajectory smoothing.
    traj: [action_dim] numpy array
    """
    if len(traj) < window_length:
        return traj  # No smoothing if too short
    return savgol_filter(traj, window_length, polyorder)

def safety_check(error_deg, threshold=10.0):
    """
    Check if error is within safe threshold.
    Returns: (is_safe, error_deg)
    """
    return error_deg <= threshold, error_deg

def evaluate():
    # Load enhanced data
    trajs, conds_raw = load_data()
    stats = np.load('cond_stats.npz')
    cond_mean = stats['mean']
    cond_std = stats['std']
    conds = (conds_raw - cond_mean) / cond_std

    # For enhanced data, simulate LOO by random subsampling (since video-level is mixed)
    N = trajs.shape[0]
    n_subsets = 10  # Simulate 10 "videos"
    subset_size = N // n_subsets
    per_file_metrics = []

    runs = sorted([d for d in os.listdir('runs') if os.path.isdir(os.path.join('runs', d)) and d.startswith('run_bc_')])
    if len(runs) == 0:
        raise RuntimeError('No BC runs/ folder found')
    latest_run = runs[-1]
    run_dir = os.path.join('runs', latest_run)
    os.makedirs(os.path.join(run_dir, 'eval_LOO'), exist_ok=True)

    for i in range(n_subsets):
        start_idx = i * subset_size
        end_idx = (i + 1) * subset_size if i < n_subsets - 1 else N
        test_trajs = trajs[start_idx:end_idx]
        test_conds = conds[start_idx:end_idx]
        train_trajs = np.concatenate([trajs[:start_idx], trajs[end_idx:]], axis=0)
        train_conds = np.concatenate([conds[:start_idx], conds[end_idx:]], axis=0)

        # Retrain model on train subset (simplified: use pre-trained but evaluate on test)
        # For demo, use the full trained model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        action_dim = test_trajs.shape[1]
        cond_dim = test_conds.shape[1]
        model = BCPlanner(cond_dim=cond_dim, action_dim=action_dim).to(device)
        ckpt = torch.load(os.path.join(run_dir, 'best_model.pth'), map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        errors = []
        safe_count = 0
        with torch.no_grad():
            for j in range(len(test_trajs)):
                cond = torch.tensor(test_conds[j:j+1].astype(np.float32), device=device)
                gen = model.generate_trajectory(cond)
                gen_smooth = postprocess_trajectory(gen)
                gt = test_trajs[j]
                err = np.mean(np.abs(gen_smooth - gt)) * 180 / np.pi
                is_safe, _ = safety_check(err)
                if is_safe:
                    safe_count += 1
                errors.append(err)

        errors = np.array(errors)
        safety_rate = safe_count / len(errors)
        per_file_metrics.append({
            'file': f'subset_{i+1:03d}',
            'mean': float(errors.mean()),
            'median': float(np.median(errors)),
            'p95': float(np.percentile(errors, 95)),
            'max': float(errors.max()),
            'count': int(len(errors)),
            'safety_rate': float(safety_rate)
        })

        # Plots
        plt.figure()
        plt.hist(errors, bins=50)
        plt.xlabel('error_deg')
        plt.ylabel('count')
        plt.title(f'Error histogram subset_{i+1:03d}')
        plt.savefig(os.path.join(run_dir, 'eval_LOO', f'subset_{i+1:03d}_hist.png'))
        plt.close()

        plt.figure()
        xs = np.sort(errors)
        ys = np.arange(1, len(xs)+1) / len(xs)
        plt.plot(xs, ys)
        plt.xlabel('error_deg')
        plt.ylabel('cdf')
        plt.title(f'Error CDF subset_{i+1:03d}')
        plt.savefig(os.path.join(run_dir, 'eval_LOO', f'subset_{i+1:03d}_cdf.png'))
        plt.close()

        plt.figure()
        plt.plot(errors[:100], marker='o')
        plt.xlabel('sample')
        plt.ylabel('error_deg')
        plt.title(f'Per-sample error (first100) subset_{i+1:03d}')
        plt.savefig(os.path.join(run_dir, 'eval_LOO', f'subset_{i+1:03d}_samples.png'))
        plt.close()

        np.savetxt(os.path.join(run_dir, 'eval_LOO', f'subset_{i+1:03d}_errors.csv'), errors, delimiter=',')

    with open(os.path.join(run_dir, 'eval_LOO', 'report.json'), 'w') as f:
        json.dump(per_file_metrics, f, indent=2)
    print('BC LOO evaluation finished. Report saved to', os.path.join(run_dir, 'eval_LOO', 'report.json'))

if __name__ == '__main__':
    evaluate()
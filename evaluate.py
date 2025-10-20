import os
import torch
import numpy as np
from diffusion_planner import DiffusionPlanner

# loads best model and cond_stats and evaluates on all data_*.npz

def load_data():
    import glob
    files = glob.glob('data_*.npz')
    trajs = []
    conds = []
    for f in files:
        d = np.load(f)
        trajs.append(d['trajectories'])
        conds.append(np.concatenate([d['poses'], d['pcs'], d['tactiles']], axis=1))
    trajs = np.concatenate(trajs, axis=0)
    conds = np.concatenate(conds, axis=0)
    return trajs, conds


def evaluate():
    # Leave-one-video-out evaluation using per-file meta
    files = sorted([f for f in os.listdir('.') if f.startswith('data_') and f.endswith('.npz')])
    if not files:
        raise RuntimeError('No data_*.npz files found')
    stats = np.load('cond_stats.npz')
    cond_mean = stats['mean']
    cond_std = stats['std']

    runs = sorted([d for d in os.listdir('runs') if os.path.isdir(os.path.join('runs', d))])
    if len(runs) == 0:
        raise RuntimeError('no runs/ folder found')
    latest_run = runs[-1]
    run_dir = os.path.join('runs', latest_run)
    os.makedirs(os.path.join(run_dir, 'eval_LOO'), exist_ok=True)

    per_file_metrics = []
    import matplotlib.pyplot as plt
    for held_out in files:
        # load model (assumes best_model.pth trained without held_out file)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # load model architecture
        # determine seq_len from one file
        d0 = np.load(files[0])
        action_dim = d0['trajectories'].shape[1]
        padded_seq_len = ((action_dim + 7) // 8) * 8  # match training padding
        model = DiffusionPlanner(cond_dim=(d0['poses'].shape[1] + d0['pcs'].shape[1] + d0['tactiles'].shape[1]), seq_len=padded_seq_len, in_channels=1).to(device)
        ckpt = torch.load(os.path.join(run_dir, 'best_model.pth'), map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        errors = []
        # evaluate only on held_out file
        d = np.load(held_out)
        trajs = d['trajectories']
        conds = np.concatenate([d['poses'], d['pcs'], d['tactiles']], axis=1)
        conds_norm = (conds - cond_mean) / cond_std
        with torch.no_grad():
            for i in range(len(trajs)):
                cond = torch.tensor(conds_norm[i:i+1].astype(np.float32), device=device)
                gen = model.generate_trajectory(cond)
                gt = trajs[i]
                err = np.mean(np.abs(gen[:action_dim] - gt)) * 180 / np.pi
                errors.append(err)
        errors = np.array(errors)
        per_file_metrics.append({'file': held_out, 'mean': float(errors.mean()), 'median': float(np.median(errors)), 'p95': float(np.percentile(errors, 95)), 'max': float(errors.max()), 'count': int(len(errors))})

        # plots
        plt.figure()
        plt.hist(errors, bins=50)
        plt.xlabel('error_deg')
        plt.ylabel('count')
        plt.title(f'Error histogram {held_out}')
        plt.savefig(os.path.join(run_dir, 'eval_LOO', f'{held_out}_hist.png'))
        plt.close()

        # CDF
        plt.figure()
        xs = np.sort(errors)
        ys = np.arange(1, len(xs)+1) / len(xs)
        plt.plot(xs, ys)
        plt.xlabel('error_deg')
        plt.ylabel('cdf')
        plt.title(f'Error CDF {held_out}')
        plt.savefig(os.path.join(run_dir, 'eval_LOO', f'{held_out}_cdf.png'))
        plt.close()

        # per-sample scatter (first 100 samples)
        plt.figure()
        plt.plot(errors[:100], marker='o')
        plt.xlabel('sample')
        plt.ylabel('error_deg')
        plt.title(f'Per-sample error (first100) {held_out}')
        plt.savefig(os.path.join(run_dir, 'eval_LOO', f'{held_out}_samples.png'))
        plt.close()

        # save per-file errors
        np.savetxt(os.path.join(run_dir, 'eval_LOO', f'{held_out}_errors.csv'), errors, delimiter=',')

    # aggregate report
    import json
    with open(os.path.join(run_dir, 'eval_LOO', 'report.json'), 'w') as f:
        json.dump(per_file_metrics, f, indent=2)
    print('LOO evaluation finished. Report saved to', os.path.join(run_dir, 'eval_LOO', 'report.json'))

if __name__ == '__main__':
    evaluate()

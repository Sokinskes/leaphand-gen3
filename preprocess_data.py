import numpy as np
from video_processor import extract_video_data
from glob import glob
from scipy.optimize import minimize

# 2. 数据扩增（添加高斯噪声）
def augment(data, n_aug=10, noise_std=0.02):
    aug = [data]
    for _ in range(n_aug):
        aug.append(data + np.random.normal(0, noise_std, data.shape))
    return np.concatenate(aug, axis=0)

# 高级数据增强：几何变换 + 噪声
def advanced_augment(trajectories, poses, pcs, tactiles, n_aug=5):
    """
    高级数据增强：包括几何变换（旋转、缩放）和噪声添加。
    trajectories: [N, action_dim] 关节角度序列
    poses: [N, 3] 物体姿态
    pcs: [N, 6144] 点云数据
    tactiles: [N, 100] 触觉数据
    """
    augmented = {
        'trajectories': [trajectories],
        'poses': [poses],
        'pcs': [pcs],
        'tactiles': [tactiles]
    }

    for _ in range(n_aug):
        # 1. 关节角度微调（模拟轻微姿态变化）
        traj_noise = trajectories + np.random.normal(0, 0.01, trajectories.shape)  # 小噪声
        # 2. 姿态变换（旋转、平移）
        pose_aug = poses.copy()
        pose_aug[:, :3] += np.random.normal(0, 0.005, (poses.shape[0], 3))  # 平移
        # 随机小角度旋转（绕Z轴）
        theta = np.random.uniform(-0.1, 0.1)  # 弧度
        rot_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                               [np.sin(theta), np.cos(theta), 0],
                               [0, 0, 1]])
        pose_aug[:, :3] = pose_aug[:, :3] @ rot_matrix.T
        # 3. 点云噪声（模拟传感器噪声）
        pcs_aug = pcs + np.random.normal(0, 0.01, pcs.shape)
        # 4. 触觉数据噪声
        tactiles_aug = tactiles + np.random.normal(0, 0.005, tactiles.shape)

        augmented['trajectories'].append(traj_noise)
        augmented['poses'].append(pose_aug)
        augmented['pcs'].append(pcs_aug)
        augmented['tactiles'].append(tactiles_aug)

    # 合并所有增强数据
    for key in augmented:
        augmented[key] = np.concatenate(augmented[key], axis=0)

    return augmented['trajectories'], augmented['poses'], augmented['pcs'], augmented['tactiles']

print("[INFO] Script started.")
print("[INFO] Searching for video files in 'videos/*.mp4' ...")
video_files = glob('videos/*.mp4')
if not video_files:
    print("[WARNING] No video files found in 'videos/' directory. Please check the path and files.")
else:
    print(f"[INFO] Found {len(video_files)} video files.")

all_trajectories, all_poses, all_pcs, all_tactiles = [], [], [], []
for idx, video_path in enumerate(video_files):
    print(f"[INFO] Processing video {idx+1}/{len(video_files)}: {video_path}")
    try:
        traj, poses, pcs, tactiles = extract_video_data(video_path)
        # 使用高级数据增强
        traj_aug, poses_aug, pcs_aug, tactiles_aug = advanced_augment(
            np.array(traj), np.array(poses), np.array(pcs), np.array(tactiles), n_aug=3
        )
        np.savez(f'data_{idx+1:03d}.npz', trajectories=traj_aug, poses=poses_aug, pcs=pcs_aug, tactiles=tactiles_aug)
        print(f"[INFO]   Saved data_{idx+1:03d}.npz with augmentation")
        # append to accumulators
        all_trajectories.append(traj_aug)
        all_poses.append(poses_aug)
        all_pcs.append(pcs_aug)
        all_tactiles.append(tactiles_aug)
    except Exception as e:
        print(f"[ERROR]   Failed to process {video_path}: {e}")




if all_trajectories:
    print("[INFO] Augmenting and saving aggregated data ...")
    trajectories = np.concatenate(all_trajectories)
    poses = np.concatenate(all_poses)
    pcs = np.concatenate(all_pcs)
    tactiles = np.concatenate(all_tactiles)
    # 使用高级增强进行全局扩增
    trajectories, poses, pcs, tactiles = advanced_augment(trajectories, poses, pcs, tactiles, n_aug=5)
    np.savez('data.npz', trajectories=trajectories, poses=poses, pcs=pcs, tactiles=tactiles)
    print(f"[INFO] Aggregated data saved to data.npz. trajectories: {trajectories.shape}, poses: {poses.shape}, pcs: {pcs.shape}, tactiles: {tactiles.shape}")
else:
    print("[WARNING] No data to save. Exiting.")

print("[INFO] Script finished.")


def merge_npz(output_npz='data_all.npz', meta_json='data_all_meta.json'):
    """Merge all data_*.npz into a single npz file and save metadata mapping.
    Metadata contains per-file start/end indices to trace samples back to original files.
    """
    import glob, json, os
    files = sorted(glob.glob('data_*.npz'))
    if not files:
        print('[WARN] No data_*.npz files found to merge.')
        return
    all_traj = []
    all_poses = []
    all_pcs = []
    all_tact = []
    meta = []
    idx = 0
    for f in files:
        d = np.load(f)
        traj = d['trajectories']
        poses = d['poses']
        pcs = d['pcs']
        tact = d['tactiles']
        n = traj.shape[0]
        all_traj.append(traj)
        all_poses.append(poses)
        all_pcs.append(pcs)
        all_tact.append(tact)
        meta.append({'file': os.path.basename(f), 'start_idx': idx, 'end_idx': idx + n - 1, 'count': n})
        idx += n
    trajectories = np.concatenate(all_traj, axis=0)
    poses = np.concatenate(all_poses, axis=0)
    pcs = np.concatenate(all_pcs, axis=0)
    tactiles = np.concatenate(all_tact, axis=0)
    np.savez(output_npz, trajectories=trajectories, poses=poses, pcs=pcs, tactiles=tactiles)
    with open(meta_json, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"[INFO] Merged {len(files)} files into {output_npz}, total samples: {trajectories.shape[0]}")
    print(f"[INFO] Metadata written to {meta_json}")


def organize_checkpoints(dest_dir=None):
    """Move .pth checkpoint files from workspace root into a timestamped archive under runs/.
    Skips files already under runs/.
    """
    import glob, os, shutil, time
    if dest_dir is None:
        dest_dir = os.path.join('runs', 'ckpt_archive_' + time.strftime('%Y%m%d_%H%M%S'))
    os.makedirs(dest_dir, exist_ok=True)
    moved = 0
    for p in glob.glob('*.pth'):
        # skip best_model or files in runs
        if os.path.commonpath([os.path.abspath(p), os.path.abspath('runs')]) == os.path.abspath('runs'):
            continue
        shutil.move(p, os.path.join(dest_dir, os.path.basename(p)))
        moved += 1
    print(f"[INFO] Moved {moved} .pth files to {dest_dir}")


if __name__ == '__main__':
    # convenience: if run directly, perform merge and organize checkpoints
    merge_npz()
    organize_checkpoints()

# 4. 关键点到 LeapHand DoF 映射（示例：最小二乘拟合）
def hand_to_leaphand_dof(hand_joints, leaphand_fk, dof_init):
    # hand_joints: (63,) 目标关键点
    # leaphand_fk: function(dof) -> (63,) LeapHand正向运动学
    def loss(dof):
        return np.linalg.norm(leaphand_fk(dof) - hand_joints)
    res = minimize(loss, dof_init, method='BFGS')
    return res.x  # 最优DoF

# 你需要实现 leaphand_fk(dof) 函数和 dof_init 初值

#!/usr/bin/env python3
"""Data Preprocessing for LeapHand Planner V4.

Converts static demonstration data into temporal sequences suitable for
Diffusion-Transformer training.
"""

import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Any
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class V4DataPreprocessor:
    """Preprocessor for V4 temporal sequence data."""

    def __init__(
        self,
        data_dir: str = "data",
        output_dir: str = "data_v4",
        seq_len: int = 10,
        stride: int = 1,
        train_split: float = 0.8
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.seq_len = seq_len
        self.stride = stride
        self.train_split = train_split

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_all_data(self) -> Dict[str, np.ndarray]:
        """Load all demonstration data."""
        logger.info("Loading demonstration data...")

        # Load metadata
        with open(self.data_dir / "data_all_meta.json", "r") as f:
            meta = json.load(f)

        all_trajectories = []
        all_poses = []
        all_pcs = []
        all_tactiles = []

        for item in tqdm(meta, desc="Loading data files"):
            file_path = self.data_dir / item["file"]
            data = np.load(file_path)

            all_trajectories.append(data["trajectories"])
            all_poses.append(data["poses"])
            all_pcs.append(data["pcs"])
            all_tactiles.append(data["tactiles"])

        # Concatenate all data
        trajectories = np.concatenate(all_trajectories, axis=0)
        poses = np.concatenate(all_poses, axis=0)
        pcs = np.concatenate(all_pcs, axis=0)
        tactiles = np.concatenate(all_tactiles, axis=0)

        logger.info(f"Loaded {len(trajectories)} total frames")
        logger.info(f"Trajectory shape: {trajectories.shape}")
        logger.info(f"Pose shape: {poses.shape}")
        logger.info(f"Point cloud shape: {pcs.shape}")
        logger.info(f"Tactile shape: {tactiles.shape}")

        return {
            "trajectories": trajectories,
            "poses": poses,
            "pcs": pcs,
            "tactiles": tactiles
        }

    def create_temporal_sequences(
        self,
        data: Dict[str, np.ndarray]
    ) -> List[Dict[str, np.ndarray]]:
        """Create temporal sequences from static data."""
        logger.info(f"Creating temporal sequences (seq_len={self.seq_len})...")

        trajectories = data["trajectories"]
        poses = data["poses"]
        pcs = data["pcs"]
        tactiles = data["tactiles"]

        sequences = []
        num_frames = len(trajectories)

        for start_idx in range(0, num_frames - self.seq_len + 1, self.stride):
            end_idx = start_idx + self.seq_len

            sequence = {
                "trajectory": trajectories[start_idx:end_idx],  # [seq_len, 63]
                "pose": poses[start_idx:end_idx],              # [seq_len, 3]
                "pc": pcs[start_idx:end_idx],                  # [seq_len, 6144]
                "tactile": tactiles[start_idx:end_idx],        # [seq_len, 100]
                "sequence_id": len(sequences),
                "start_frame": start_idx,
                "end_frame": end_idx
            }

            sequences.append(sequence)

        logger.info(f"Created {len(sequences)} temporal sequences")
        return sequences

    def split_train_val(
        self,
        sequences: List[Dict[str, np.ndarray]]
    ) -> tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
        """Split sequences into train and validation sets."""
        num_sequences = len(sequences)
        num_train = int(num_sequences * self.train_split)

        # Shuffle sequences
        np.random.seed(42)
        indices = np.random.permutation(num_sequences)
        sequences = [sequences[i] for i in indices]

        train_sequences = sequences[:num_train]
        val_sequences = sequences[num_train:]

        logger.info(f"Train sequences: {len(train_sequences)}")
        logger.info(f"Val sequences: {len(val_sequences)}")

        return train_sequences, val_sequences

    def save_sequences(
        self,
        sequences: List[Dict[str, np.ndarray]],
        split_name: str
    ):
        """Save sequences to disk."""
        output_file = self.output_dir / f"{split_name}_sequences.npz"

        # Convert to arrays
        trajectory_data = np.array([seq["trajectory"] for seq in sequences])
        pose_data = np.array([seq["pose"] for seq in sequences])
        pc_data = np.array([seq["pc"] for seq in sequences])
        tactile_data = np.array([seq["tactile"] for seq in sequences])

        # Save as compressed numpy arrays
        np.savez_compressed(
            output_file,
            trajectories=trajectory_data.astype(np.float32),
            poses=pose_data.astype(np.float32),
            pcs=pc_data.astype(np.float32),
            tactiles=tactile_data.astype(np.float32)
        )

        logger.info(f"Saved {len(sequences)} sequences to {output_file}")

        # Save metadata
        meta_file = self.output_dir / f"{split_name}_meta.json"
        metadata = {
            "num_sequences": len(sequences),
            "seq_len": self.seq_len,
            "trajectory_shape": trajectory_data.shape[1:],  # [seq_len, 63]
            "pose_shape": pose_data.shape[1:],              # [seq_len, 3]
            "pc_shape": pc_data.shape[1:],                  # [seq_len, 6144]
            "tactile_shape": tactile_data.shape[1:],        # [seq_len, 100]
            "data_types": {
                "trajectories": "float32",
                "poses": "float32",
                "pcs": "float32",
                "tactiles": "float32"
            }
        }

        with open(meta_file, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {meta_file}")

    def preprocess(self):
        """Run full preprocessing pipeline."""
        logger.info("Starting V4 data preprocessing...")

        # Load data
        data = self.load_all_data()

        # Create sequences
        sequences = self.create_temporal_sequences(data)

        # Split train/val
        train_sequences, val_sequences = self.split_train_val(sequences)

        # Save processed data
        self.save_sequences(train_sequences, "train")
        self.save_sequences(val_sequences, "val")

        # Create summary
        summary = {
            "original_frames": len(data["trajectories"]),
            "sequence_length": self.seq_len,
            "stride": self.stride,
            "train_sequences": len(train_sequences),
            "val_sequences": len(val_sequences),
            "total_sequences": len(sequences),
            "output_dir": str(self.output_dir)
        }

        summary_file = self.output_dir / "preprocessing_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Preprocessing complete! Summary saved to {summary_file}")
        logger.info(f"Train sequences: {len(train_sequences)}")
        logger.info(f"Val sequences: {len(val_sequences)}")

        return summary


def main():
    """Main preprocessing function."""
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess data for LeapHand V4")
    parser.add_argument("--data_dir", type=str, default="data",
                       help="Input data directory")
    parser.add_argument("--output_dir", type=str, default="data_v4",
                       help="Output directory for processed data")
    parser.add_argument("--seq_len", type=int, default=10,
                       help="Sequence length for temporal modeling")
    parser.add_argument("--stride", type=int, default=1,
                       help="Stride for sequence creation")
    parser.add_argument("--train_split", type=float, default=0.8,
                       help="Train/validation split ratio")

    args = parser.parse_args()

    preprocessor = V4DataPreprocessor(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seq_len=args.seq_len,
        stride=args.stride,
        train_split=args.train_split
    )

    summary = preprocessor.preprocess()
    print("\nPreprocessing Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
# LeapHand Trajectory Planner

A behavioral cloning system for generating dexterous manipulation trajectories for the LeapHand robotic gripper using AI-generated video data.

## Features

- **Behavioral Cloning**: MLP-based trajectory prediction from multimodal inputs
- **Data Augmentation**: Advanced augmentation with geometric transforms and noise
- **Real-time Inference**: Fast trajectory generation (<1ms) for real-time control
- **Safety Checks**: Built-in safety validation with configurable thresholds
- **Post-processing**: Savitzky-Golay smoothing for trajectory refinement

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd leaphandgen3
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
leaphandgen3/
├── leap_hand_planner/          # Main package
│   ├── models/                 # Model definitions
│   │   ├── bc_planner.py      # BC Planner model
│   ├── data/                   # Data processing
│   │   ├── loader.py          # Data loading utilities
│   ├── utils/                  # Utility functions
│   │   ├── trajectory.py      # Trajectory processing
│   └── config/                 # Configuration
│       ├── default.yaml       # Default configuration
├── data/                       # Data files
├── runs/                       # Experiment results
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## Usage

### Training

```bash
python -m leap_hand_planner.train_bc
```

### Evaluation

```bash
python -m leap_hand_planner.evaluate_bc
```

### Inference

```python
from leap_hand_planner.models.bc_planner import BCPlanner
import torch

# Load model
model = BCPlanner(cond_dim=6247, action_dim=63)
model.load_state_dict(torch.load('path/to/model.pth'))
model.eval()

# Generate trajectory
condition = torch.randn(1, 6247)  # Your condition vector
trajectory = model.generate_trajectory(condition)
```

## Configuration

Edit `leap_hand_planner/config/default.yaml` to modify training parameters, model architecture, and evaluation settings.

## Data Format

- **Trajectories**: [N, 63] - Joint angles in radians
- **Poses**: [N, 3] - Object pose (x, y, z)
- **Point Clouds**: [N, 6144] - Flattened point cloud data
- **Tactile**: [N, 100] - Tactile sensor readings

## Results

Current BC model achieves:
- Mean error: 0.6-1.1°
- Median error: 0.4-0.6°
- 95th percentile: 0.75-2.8°
- Safety rate: 100% (within 10° threshold)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license here]
# DADP: Dynamics-Aligned Diffusion Planning for Offline RL

This repository contains the source code for the paper **"Dynamics-Aligned Diffusion Planning for Offline RL: A Unified Framework with Forward and Inverse Guidance"** (accepted by TMLR 2026).

## Overview

DADP is a unified diffusion-based planning framework that leverages learned dynamics models for efficient trajectory optimization in offline reinforcement learning. The framework incorporates both forward and inverse guidance mechanisms to align the diffusion process with environment dynamics.

**Note:** This implementation is built upon the [Diffuser](https://github.com/jannerm/diffuser) codebase and can be integrated or plugged into existing diffusion-based planning pipelines.

## Installation

### Prerequisites
- Python 3.8
- CUDA 11.1 (for GPU support)
- MuJoCo 2.0

### Setup with Conda

```bash
conda env create -f environment.yml
conda activate dadp
```

### Setup with pip

```bash
pip install -r requirements.txt
```

## Project Structure

```
DADP/
├── config/              # Configuration files
├── dadp/                # Main package
│   ├── datasets/        # Dataset loading and preprocessing
│   ├── environments/    # Custom environment definitions
│   ├── models/          # Diffusion and dynamics models
│   ├── sampling/        # Sampling and planning algorithms
│   └── utils/           # Utility functions
├── dynamics_model/      # Dynamics model training
├── plotting/            # Visualization scripts
└── scripts/             # Training and evaluation scripts
```

## Usage

### Training Dynamics Model

Train the inverse dynamics model (VAE-based) for guidance:

```bash
python dynamics_model/main.py
```

The dynamics model will be saved in `inverse_dynamics_model/models/` with training loss plots.

### Training Diffusion Model

```bash
python scripts/train.py --dataset hopper-medium-expert-v2
```

### Training Value Function

```bash
python scripts/train_values.py --dataset hopper-medium-expert-v2
```

### Planning and Evaluation

```bash
python scripts/plans_inverse.py --dataset hopper-medium-expert-v2
```

## Supported Environments

- HalfCheetah (all variants)
- Hopper (all variants)
- Walker2d (all variants)
- maze2d (all variants)

All environments use D4RL datasets with versions: random, medium, medium-replay, medium-expert, expert.

## Configuration

Edit configuration files in `config/locomotion.py` to customize:
- Model architecture (horizon, dimensions, attention)
- Training hyperparameters (learning rate, batch size)
- Planning parameters (guidance scale, timesteps)

## Results

Results will be saved in the `logs/` directory, including:
- Model checkpoints
- Training metrics
- Evaluation videos

## Citation

If you use this code in your research, please cite:

```bibtex
@article{wang2026dadp,
  title={Dynamics-Aligned Diffusion Planning for Offline RL: A Unified Framework with Forward and Inverse Guidance},
  author={Wang, Zihao and Jiang, Ke and Tan, Xiaoyang},
  journal={Transactions on Machine Learning Research},
  year={2026},
  note={Accepted}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This work builds upon:
- [D4RL](https://github.com/Farama-Foundation/d4rl) for offline RL datasets
- [Diffuser](https://github.com/jannerm/diffuser) for diffusion-based planning baseline

The codebase is built on top of the Diffuser framework, allowing seamless integration with diffusion-based trajectory optimization methods.

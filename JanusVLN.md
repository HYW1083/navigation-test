# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

JanusVLN is a Vision-Language Navigation (VLN) framework featuring dual implicit memory that decouples semantics and spatiality. It's built on Qwen2.5-VL-7B and VGGT-1B models and uses Habitat-sim for 3D environment simulation.

The project trains navigation agents on R2R, RxR, and ScaleVLN datasets for the Matterport3D (MP3D) and HM3D scene datasets.

## Environment Setup

```bash
# Create environment
conda create -n janusvln python=3.9 -y && conda activate janusvln

# Install habitat-sim (version 0.2.4 is critical)
conda install habitat-sim==0.2.4 withbullet headless -c conda-forge -c aihabitat

# Install habitat-lab from source (v0.2.4)
git clone --branch v0.2.4 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -e habitat-lab
pip install -e habitat-baselines
cd ..

# Install PyTorch (CUDA 12.4)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install dependencies and package
pip install -r requirements.txt
pip install -e .
```

## Data Structure

The project expects a specific data directory structure:
- `data/scene_datasets/mp3d/` - Matterport3D scenes for R2R/RxR
- `data/scene_datasets/hm3d/` - HM3D scenes for ScaleVLN
- `data/datasets/r2r/`, `data/datasets/rxr/`, `data/datasets/scalevln/` - VLN-CE episode data
- `data/trajectory_data/` - Pre-collected observation-action trajectories
- `data/dagger_data/` - DAgger collected data

## Key Commands

### Build Dataset
```bash
# Base dataset (R2R-CE + RxR-CE only)
python create_data/create_data.py

# Extra dataset (includes DAgger + ScaleVLN)
python create_data/create_data.py --use_extra_data
```

After building datasets, configure paths in `src/qwen_vl/data/__init__.py` by uncommenting and setting the TRAIN_R2R_RxR and TRAIN_R2R_RxR_EXTRA dataset entries.

### Training

```bash
# Base model training (R2R-CE + RxR-CE)
bash scripts/train.sh

# Extra model training (with DAgger + ScaleVLN)
bash scripts/train_extra.sh
```

Training uses torchrun with automatic GPU detection and DeepSpeed ZeRO-3.

### DAgger Data Collection

```bash
# Edit scripts/dagger.sh to set:
# - DAGGER_DATASET (R2R or RxR)
# - DAGGER_DATA_PATH
# - DAGGER_GT_ANNOTATIONS_PATH
# - MID_RUN_NAME (checkpoint to use)

bash scripts/dagger.sh
```

### Evaluation

```bash
# Edit scripts/evaluation.sh to set:
# - CHECKPOINT (model path)
# - CONFIG (config/vln_r2r.yaml or config/vln_dagger.yaml)
# - OUTPUT_PATH

bash scripts/evaluation.sh
```

Runs on 8 GPUs by default with multi-process evaluation.

## Architecture

### Core Components

**src/qwen_vl/model/**
- `modeling_qwen2_5_vl.py` - Main model implementation (Qwen2_5_VLForConditionalGenerationForJanusVLN)
- `vggt/` - VGGT (Vision-Guided Graph Transformer) spatial memory module
- `loss.py` - Custom loss functions for VLN training

**src/qwen_vl/data/**
- `data_qwen.py` - Data loading and preprocessing pipeline
- `draw_marker.py` - Trajectory visualization with markers
- `rope2d.py` - 2D rotary position embeddings

**src/qwen_vl/train/**
- Training loop implementations

**src/habitat_extensions/**
- `measures.py` - Custom evaluation metrics for VLN (Success, SPL, Oracle metrics)
- `maps.py` - Top-down map generation and visualization utilities

**Entry Points:**
- `src/evaluation.py` - Inference and evaluation in Habitat environments
- `src/dagger.py` - DAgger data collection using trained models

### Model Architecture

JanusVLN extends Qwen2.5-VL with dual implicit memory:
1. **Semantic Memory**: Language-conditioned visual understanding through Qwen2.5-VL backbone
2. **Spatial Memory**: 3D spatial reasoning through VGGT module

The model takes trajectory history (up to 8 frames) and instruction text, processes them through dual memory streams, and outputs navigation actions.

### Configuration System

Uses Hydra/OmegaConf for habitat configuration:
- `config/vln_r2r.yaml` - R2R dataset configuration (MP3D scenes, 640x480 RGB-D)
- `config/vln_dagger.yaml` - DAgger configuration

Key settings:
- Forward step: 0.25m
- Turn angle: 15 degrees
- Episode length: 500 steps
- Success threshold: 3.0m

### DeepSpeed Configuration

Training uses DeepSpeed ZeRO for memory efficiency:
- `scripts/zero0.json` - No sharding
- `scripts/zero2.json` - Optimizer state sharding
- `scripts/zero3.json` - Full parameter + optimizer sharding (default)

## Training Configuration

Default hyperparameters in `scripts/train.sh`:
- Batch size: 1 per device, gradient accumulation: 8
- Learning rates: LLM 2e-5, projector 1e-5, vision 1e-6
- Max length: 163,840 tokens
- Video frames: 4-8 frames, dynamic resolution
- Optimizer: AdamW with cosine schedule
- Warmup: 3% of steps
- Checkpointing: Every 1000 steps

## Model Outputs

Models are saved to:
- `./JanusVLN_Base/` - Base model checkpoints
- `./JanusVLN_Extra/` - Extra model checkpoints

Evaluation results go to the `evaluation/` directory with optional video outputs.

## Important Notes

1. **Version Requirements**: Habitat-sim 0.2.4 is critical - newer versions may have compatibility issues
2. **Dataset Configuration**: After running `create_data.py`, manually configure dataset paths in `src/qwen_vl/data/__init__.py`
3. **Multi-GPU Training**: Scripts auto-detect GPUs using `nvidia-smi --list-gpus | wc -l`
4. **Distributed Ports**: Random port selection (20000-29999) to avoid conflicts
5. **Habitat Logging**: Scripts set `MAGNUM_LOG=quiet HABITAT_SIM_LOG=quiet` to reduce noise
6. **Model Zoo**: Pre-trained models available on ModelScope (misstl/JanusVLN_Base, misstl/JanusVLN_Extra)

## Development Workflow

1. Set up environment and download scene datasets + VLN episodes
2. Download trajectory data from ModelScope or collect your own
3. Run `create_data.py` to build training datasets
4. Configure dataset paths in `src/qwen_vl/data/__init__.py`
5. Train base model with `scripts/train.sh`
6. Collect DAgger data with trained base model
7. Train extra model with DAgger + ScaleVLN data
8. Evaluate on validation sets using `scripts/evaluation.sh`

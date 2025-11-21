# Project Structure Guide

## Overview
This project uses **Graph Neural Networks (GNNs)** to predict molecular properties from the **QM9 dataset**. The code is organized with a modular structure using **Hydra** for configuration management and **Weights & Biases** for experiment tracking.

---

## Project Structure

```
02456-g82-dl-project/
│
├── configs/                      # Configuration files (YAML)
│   ├── dataset/
│   │   └── qm9.yaml              # Dataset configuration
│   ├── experiment/               # Holds experiment configurations
│   │   ├── my-experiment.yaml    # Specific experiment configuration
│   │   └── ...
│   ├── logger/                   # Holds logger configurations
│   │   └── wandb.yaml            # Specific logger configuration
│   ├── model/                    # Holds model architecture configurations
│   │   ├── gcn.yaml              # Configuration of specific model or model ensemble
│   │   └── ...
│   ├── optimizer/                # Holds optimizer configurations
│   │   ├── adamw.yaml            # Specific optimizer configuration
│   │   └── sgd.yaml
│   ├── run.yaml                  # Main config that ties everything together
│   ├── scheduler/                # Holds scheduler configurations
│   │   ├── base_scheduler.yaml   # Base configuration for all the schedulers
│   │   ├── sequential.yaml       # Specific scheduler configuration
│   │   └── ...
│   └── trainer/                  # Holds training configurations
│       ├── n-cps.yaml            # Configuration of specific trainer
│       └── ...
│─ scripts/                       # Holds utility scripts for running the project
│   └── ...
└── src/                          # Source code
    ├── dataset_utils.py          # Dataset utilities
    ├── logger.py
    ├── models/                   # Holds different model architectures (GCN, etc.)
    │   ├── initial.py            # Original model architecture
    │   └── ...
    ├── model_utils.py            # Model setup utilities
    ├── qm9.py                    # Dataset loading and preparation
    ├── qm9_utils.py              # Dataset utilities
    ├── run.py                    # Main entry point
    ├── trainers/                 # Holds different trainers
    │   ├── ncps.py               # Specific trainer
    │   └── ...
    └── utils.py
```

---

## How the Files Relate

### Execution Flow:
1. **`run.py`** (orchestrator) loads all configs from `configs/`
2. Uses **`utils.py`** to set random seeds
3. Instantiates **`logger.py`** (WandB) to track experiments
4. Instantiates **`qm9.py`** (data module) to load and split data
5. Instantiates configured model(s) from **`models`** package (for example, `models/initial.py#GCN`)
6. Instantiates configured trainer from **`trainers`** package (training loop)
7. Calls `trainer.train()` which uses the data module and logs metrics

### Configuration Flow:
- **YAML files** define all settings (hyperparameters, paths, etc.)
- **Hydra** loads and merges these configs
- **`run.py`** uses `hydra.utils.instantiate()` to create objects based on configs

---

## Key Files Explained

### 1. `src/run.py` - Main Entry Point
**What it does**: Orchestrates the entire experiment.

**Key steps**:
- Loads configuration files
- Selects device (CPU/GPU)
- Initializes logger, data, model, and trainer
- Runs training

**When to modify**: Rarely. Most changes should be in configs or other files.

---

### 2. `src/models/` - Model Architectures organized into separate files
**What it does**: Defines neural network architectures.

**Current default model**: is defined in `run.yaml` defaults list, under `model`. For `modelxyz` in `run.yaml`, look up the `__target__` in corresponding `configs/model/modelxyz.yaml` (currently that resolves to `models.initial.GCN`).

**To add a new model**:
1. Add a new file under `models` folder, for example `my-new-arch.py`:
   ```python
   class MyNewGNN(torch.nn.Module):
       def __init__(self, num_node_features, hidden_channels=64):
           super().__init__()
           # Define layers here
       
       def forward(self, data):
           # Define forward pass
           return x
   ```

2. Create a new config file `configs/model/mynewgnn.yaml`:
   ```yaml
   name: mynewgnn
   
   init:
     _target_: models.my-new-arch.MyNewGNN
     num_node_features: 11
     hidden_channels: 128
   ```

3. Run with: `python src/run.py model=mynewgnn`

---

### 3. `src/trainers/` - Trainer methods organized into separate files
**What it does**: Defined different ways for handling training and validation logic.

**Current default trainer**: is defined in `run.yaml` defaults list, under `trainer`. For `trainerxyz` in `run.yaml`, look up the `__target__` in corresponding `trainer/trainerxyz.yaml` (currently that resolves to `trainers.supervised.SupervisedEnsemble`).

**Key methods**:
- `train()`: Main training loop
- `validate()`: Computes validation metrics (MSE)

**To modify training logic**: Edit one of the files in this directory (e.g., modify how loss is calculated, change optimization strategy), or create a new one.

---

### 4. `src/qm9.py` - Dataset Module
**What it does**: Loads and prepares the QM9 molecular dataset.

**Key features**:
- Splits data into train/validation/test
- Creates dataloaders
- Handles batch sizes

**To change data setup**: Edit this file or the config `configs/dataset/qm9.yaml`.

---

### 5. `src/logger.py` - Experiment Tracking
**What it does**: Logs metrics to Weights & Biases.

**Key methods**:
- `init_run()`: Starts a new experiment run
- `log_dict()`: Logs metrics (loss, MSE, etc.)

**To disable logging**: Set `disable: true` in `configs/logger/wandb.yaml`.

---

### 6. `src/utils.py` - General Utilities
**What it does**: Helper functions used across the project.

**Examples**:
- `seed_everything()`: Sets random seeds
- `validate_models()`: Validation helper
- `save_results()`: Saves experiment results

---

### 7. Configuration Files (`configs/`)
**What they do**: Store all experiment settings in YAML format.

**Why use configs?**
- Easy to change settings without modifying code
- Reproducible experiments
- Easy hyperparameter tuning

---

## Where to Make Changes

These steps described here are for making changes to the default parameters for each of the component type and architecture/method, and should at the end contain the current optimized version of the parameters. Only commit these changes if they have been determined as current optimal settings. Otherwise use experiments (see sections below) or revert the changes before comitting to the repository.

### To Change Hyperparameters:

#### **Optimizer**:
Edit `configs/run.yaml` to select different type of optimizer:
```yaml
defaults:
  - dataset: qm9
  - trainer: supervised-ensemble
  - optimizer: adamw  # Change this
  - scheduler: step
  - logger: wandb
  - model: gcn
  - _self_
```

#### **Learning Rate**:
Edit learning rate of the chosen optimizer in its config, for example `configs/optimizer/adamw.yaml`:
```yaml
init:
  _target_: torch.optim.AdamW
  lr: 0.001  # Change this
  weight_decay: 0.005
```

#### **Scheduler**:
Edit `configs/run.yaml` to select different type of scheduler:
```yaml
defaults:
  - dataset: qm9
  - trainer: supervised-ensemble
  - optimizer: adamw
  - scheduler: step  # Change this
  - logger: wandb
  - model: gcn
  - _self_
```

Edit parameters of a specific scheduler in its config, for example `configs/scheduler/step.yaml`:
```yaml
defaults:
  - base_scheduler

init:
  _target_: torch.optim.lr_scheduler.StepLR
  step_size: 1
  gamma: 0.975  # Change this
```


#### **Batch Size, Data Splits**:
Edit `configs/dataset/qm9.yaml`:
```yaml
batch_size_train: 100     # Change this
splits: [0.72, 0.08, 0.1, 0.1]  # Change data splits
```

#### **Model Architecture (Hidden Channels, Layers)**:
Edit `configs/model/gcn.yaml`:
```yaml
init:
  num_node_features: 11
  hidden_channels: 64     # Change this (or add more parameters)
```

Then update the model (for example, `src/models/initial.py`) to accept the new parameters:
```python
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_layers=2):
        # Use num_layers parameter
```

---

### To Create a More Advanced GNN:

#### **Option 1: Modify Existing Models package**
Edit the model package (for example, `src/models/initial.py`) to add a new model and:
- More layers
- Different GNN layers (GAT, GraphSAGE, GIN)
- Dropout, batch normalization
- Residual connections

Example:
```python
from torch_geometric.nn import GATConv, GINConv, global_add_pool

# ...
# Existing models in the file, e.g.
# class GCN(torch.nn.Module):
# ...

class AdvancedGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_channels=64, num_layers=3, dropout=0.1):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(num_node_features, hidden_channels, heads=4))
        
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * 4, hidden_channels, heads=4))
        
        self.dropout = dropout
        self.linear = torch.nn.Linear(hidden_channels * 4, 1)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = global_add_pool(x, batch)
        x = self.linear(x)
        return x
```

#### **Option 2: Create New Model File**
1. Create `src/models/advanced_gnn.py`
2. Add new config `configs/model/advanced_gnn.yaml`
3. Update `_target_` in config to point to new model

---

### To Use Different Datasets:

1. Create a new dataset module (similar to `qm9.py`)
2. Create a new config file in `configs/dataset/`
3. Run with: `python src/run.py dataset=new_dataset`

---

## Running Experiments

### Basic Run:
```bash
python src/run.py
```

### Override Configurations:
```bash
# Change model
python src/run.py model=gcn

# Change hyperparameters
python src/run.py optimizer.init.lr=0.01

# Change target property
python src/run.py dataset.target=5

# Change number of epochs
python src/run.py trainer.train.total_epochs=500

# Multiple overrides
python src/run.py model=gcn optimizer.init.lr=0.001 dataset.batch_size_train=128
```

---

## Hyperparameter Tuning with Weights & Biases

### Manual Tuning:
Run multiple experiments with different settings:
```bash
python src/run.py optimizer.init.lr=0.001
python src/run.py optimizer.init.lr=0.01
python src/run.py optimizer.init.lr=0.1
```

### Automated Tuning (WandB Sweeps):
1. Create a sweep config file `sweep.yaml`:
   ```yaml
   program: src/run.py
   method: bayes
   metric:
     name: val_MSE
     goal: minimize
   parameters:
     optimizer.init.lr:
       min: 0.0001
       max: 0.1
     model.init.0.hidden_channels:
       values: [32, 64, 128, 256]
   ```

2. Initialize sweep:
   ```bash
   wandb sweep sweep.yaml
   ```

3. Run agents:
   ```bash
   wandb agent <sweep-id>
   ```

---

## Tips for Improving Performance

### 1. **Architecture Changes** (in `src/models/`):
- Add more layers
- Use different GNN types (GAT, GIN, GraphSAGE)
- Add skip connections
- Use attention mechanisms
- Add batch normalization or layer normalization

### 2. **Training Improvements** (in `configs/trainer/`):
- Try different optimizers (Adam, AdamW, SGD)
- Adjust learning rate and scheduler
- Add gradient clipping
- Try different loss functions

### 3. **Data Improvements** (in `src/qm9.py` or `configs/dataset/`):
- Change data splits
- Add data augmentation
- Use different molecular features
- Try different batch sizes

### 4. **Regularization** (in `src/models/` or config):
- Add dropout
- Increase weight decay
- Use early stopping

---

## Troubleshooting

### GPU Not Detected:
- Check device selection in `run.py`
- For Mac M4: Ensure MPS is enabled
- For HPC: Load CUDA module first

### Config Errors:
- Check YAML syntax
- Ensure `_target_` points to correct class
- Verify all required parameters are provided

### Training Issues:
- Check validation MSE in WandB dashboard
- Reduce learning rate if training is unstable
- Check for NaN values in loss

---

## Summary

**To change hyperparameters**: Edit YAML files in `configs/`

**To create new models**: Edit files in `src/models/` and create new config

**To modify training**: Edit files in `src/trainers/`

**To track experiments**: Use Weights & Biases dashboard

**To run experiments**: Use `python src/run.py` with config overrides

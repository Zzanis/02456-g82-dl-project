# Semi Supervisde Learnings in GNN for Drug Discovery 

This project builds upon an existing semi-supervised Graph Neural Networks (GNNs) on the QM9 dataset using PyTorch and PyTorch Geometric from <[gnn_intro](https://github.com/tirsgaard/gnn_intro)>, extending it with semi-supervised training using established methods Mean Teacher and n-CPS.

## Installation

To run this project, you need to install the required Python packages. You can install them using pip:

```bash
# It is recommended to install PyTorch first, following the official instructions
# for your specific hardware (CPU or GPU with a specific CUDA version).
# See: https://pytorch.org/get-started/locally/

# For example, for a recent CUDA version:
# pip install torch torchvision torchaudio

# Or for CPU only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# After installing PyTorch, install PyTorch Geometric.
# The exact command depends on your PyTorch and CUDA versions.
# Please refer to the PyTorch Geometric installation guide:
# https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html

# Example for PyTorch 2.7 and CUDA 11.8
# pip install torch_geometric

# Then, install the other required packages:
pip install hydra-core omegaconf wandb pytorch-lightning numpy tqdm
```

## How to Run

The main entry point for this project is `src/run.py`. It uses `hydra` for configuration management. Hydra is a broadly used and highly respected so I recommend using it. You can find a guide to it here https://medium.com/@jennytan5522/introduction-to-hydra-configuration-for-python-646e1dd4d1e9.

To run the code, execute the following command from the root of the project:

```bash
python src/run.py
```

You can override the default configuration by passing arguments from the command line. For example, to use a different model configuration:

```bash
python src/run.py model=gcn
```

The configuration files are located in the `configs/` directory.

### Using experiments

Use experiments to document multiple different configurations of interest for repeatable runs.

Create your own experiment config file (for example, `my-experiment.yaml`) under `src/config/experiments`, and in it specify the overrides for model, trainer etc as needed (see [Hydra docs](https://hydra.cc/docs/patterns/configuring_experiments/) for more details).

Run the experiment by appending the experiment name (file name without the extension):

```bash
python src/run.py +experiment=my-experiment
```

Experiment can be combined with other command-line overrides - these will override whatever is in the experiment as well:

```bash
python src/run.py +experiment=my-experiment trainer.num_models=3
```





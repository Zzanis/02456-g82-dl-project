from itertools import chain
import hydra
import torch
from omegaconf import OmegaConf

from utils import seed_everything


@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg):
    # Print the full configuration for transparency
    print(OmegaConf.to_yaml(cfg))

    # Select device (GPU or CPU)
    if cfg.device in ["unset", "auto"]:
        if torch.backends.mps.is_available():
            device = torch.device("mps")  # Use Apple Silicon GPU
        elif torch.cuda.is_available():
            device = torch.device("cuda")  # Use NVIDIA GPU (on HPC)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(cfg.device)

    # Set random seeds for reproducibility
    seed_everything(cfg.seed, cfg.force_deterministic)

    # Initialize the experiment logger (e.g., Weights & Biases)
    logger = hydra.utils.instantiate(cfg.logger)
    hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger.init_run(hparams)

    # Load the dataset (QM9 molecular data)
    dm = hydra.utils.instantiate(cfg.dataset.init)

    # Create ensemble of models
    num_models = cfg.trainer.num_models
    models = []

    for i in range(num_models):
        # Create the model (GNN) and move it to the selected device
        gnn_model = hydra.utils.instantiate(cfg.model.init).to(device)

        # Optionally compile the model for faster execution
        if cfg.compile_model:
            gnn_model = torch.compile(gnn_model)

        models.append(gnn_model)
    
    # Initialize the trainer with model, logger, data, and device
    trainer = hydra.utils.instantiate(cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device)

    # Train the model and collect results
    results = trainer.train(**cfg.trainer.train)
    if results is not None:
        results = torch.Tensor(results)



if __name__ == "__main__":
    main()

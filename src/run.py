import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from utils import seed_everything


@hydra.main(
    config_path="../configs/",
    config_name="run.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
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

    models = []
    # Handle one or more instances of the same model, where cfg.model.init is a single instance
    if isinstance(cfg.model.init, DictConfig):
        # Default is 1 instance
        num_models = cfg.trainer.get("num_models", 1)
        print(f"Will initialise {num_models} {cfg.model.name} models")
        for i in range(num_models):
            # Create the model (GNN) and move it to the selected device
            gnn_model = hydra.utils.instantiate(cfg.model.init).to(device)
            models.append(gnn_model)
    else:
        # Handle instances of models with different architectures, where cfg.model.init is a list
        for model_init in cfg.model.init:
            model = hydra.utils.instantiate(model_init).to(device)
            print(f"arch: {model}")
            # logger.log("architecture", model)
            models.append(model)

    # Optionally compile the model for faster execution
    if cfg.compile_model:
        for i, model in enumerate(models):
            models[i] = torch.compile(model)

    # Initialize the trainer with models, logger, data, and device
    trainer = hydra.utils.instantiate(
        cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device
    )

    # Train the model and collect results
    results = trainer.train(**cfg.trainer.train)
    if results is not None:
        results = torch.Tensor(results)


if __name__ == "__main__":
    main()

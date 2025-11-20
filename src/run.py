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

    models = []
    for model_init in cfg.model.init:
        model = hydra.utils.instantiate(model_init).to(device)
        print(f"arch: {model}")
        # logger.log("architecture", model)

        # Optionally compile the model for faster execution
        if cfg.compile_model:
            model = torch.compile(model)
        models.append(model)

    # Setup optimizer and scheduler here to avoid dealing with nested partials
    # in case of nested schedulers
    all_params = [p for m in models for p in m.parameters()]
    optimizer = hydra.utils.instantiate(cfg.optimizer.init, params=all_params)
    # Deal with nested schedulers that require optimizer as 1st positional arg
    if cfg.scheduler.sequential:
        schedulers = [
            hydra.utils.instantiate(scfg, optimizer)
            for scfg in cfg.scheduler.schedulers
        ]
        scheduler = hydra.utils.instantiate(cfg.scheduler.init, optimizer, schedulers)
    else:
        scheduler = hydra.utils.instantiate(cfg.scheduler.init, optimizer)

    # Initialize the trainer
    trainer = hydra.utils.instantiate(
        cfg.trainer.init,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        models=models,
        datamodule=dm,
        logger=logger,
    )

    # Train the model and collect results
    results = trainer.train(**cfg.trainer.train)
    if results is not None:
        results = torch.Tensor(results)



if __name__ == "__main__":
    main()

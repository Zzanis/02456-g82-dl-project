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

    # Create the model (GNN) and move it to the selected device
    model = hydra.utils.instantiate(cfg.model.init).to(device)

    # Optionally compile the model for faster execution
    if cfg.compile_model:
        model = torch.compile(model)
    models = [model]
    
    # cfg.trainer.init.lambda_max = cfg.trainer.train.lambda_max
    # cfg.trainer.init.total_epochs = cfg.trainer.train.total_epochs
    # # Initialize the trainer with model, logger, data, and device
    trainer = hydra.utils.instantiate(cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device)

    # Train the model and collect results
    results = trainer.train(**cfg.trainer.train)
    if results is not None:
        results = torch.Tensor(results)

    # if hasattr(trainer, "test"):
    #     test_results = trainer.test()

    #     print("\n==============================")
    #     print("   FINAL TEST SET RESULTS")
    #     print("==============================")
    #     print(test_results)

    #     # --- Log test results to W&B ---
    #     if isinstance(test_results, dict):
    #         logger.log_metrics(test_results, step=0)
    #     elif isinstance(test_results, (list, tuple)) and isinstance(test_results[0], dict):
    #         for d in test_results:
    #             logger.log_metrics(d, step=0)
    #     else:
    #         print("WARNING: Test results could not be logged to wandb.")

    # else:
    #     print("WARNING: trainer has no .test() method")


if __name__ == "__main__":
    main()

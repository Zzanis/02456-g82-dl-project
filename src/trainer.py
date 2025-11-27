from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

class SupervisedEnsemble:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        patience=20,
        min_delta=0.001,
    ):
        self.device = device
        self.models = models

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_models_state = None

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []
        
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # Ensemble prediction
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)
                
                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def train(self, total_epochs, validation_interval):
        #self.logger.log_dict()
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()
            supervised_losses_logged = []
            for x, targets in self.train_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                # Supervised loss
                supervised_losses = [self.supervised_criterion(model(x), targets) for model in self.models]
                supervised_loss = sum(supervised_losses)
                supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))  # type: ignore
                loss = supervised_loss
                loss.backward()  # type: ignore
                # adding gradient clipping - fixed to clip all models in ensemble
                for model in self.models:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                self.optimizer.step()
            self.scheduler.step()
            supervised_losses_logged = np.mean(supervised_losses_logged)

            summary_dict = {
                "supervised_loss": supervised_losses_logged,
                "learning_rate": self.optimizer.param_groups[0]['lr'],  
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)

                # Early stopping check
                current_val_loss = val_metrics["val_MSE"]
                if current_val_loss < self.best_val_loss - self.min_delta:
                    # Improvement detected
                    self.best_val_loss = current_val_loss
                    self.patience_counter = 0
                    # Save best model states
                    self.best_models_state = [model.state_dict() for model in self.models]
                    summary_dict["early_stop_best"] = True
                else:
                    # No improvement
                    self.patience_counter += 1
                    summary_dict["early_stop_patience"] = self.patience_counter
                
                # Check if patience exceeded
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}. Best val_MSE: {self.best_val_loss:.4f}")
                    # Restore best model weights
                    if self.best_models_state is not None:
                        for model, state in zip(self.models, self.best_models_state):
                            model.load_state_dict(state)
                        print("Restored best model weights")
                    break
            self.logger.log_dict(summary_dict, step=epoch)


class SemiSupervisedCPS:
    """
    Semi-Supervised Cross Pseudo Supervision (CPS) Trainer.
    
    Implements the CPS method for graph-level regression tasks.
    Two networks with identical architecture train jointly:
    - Network A produces pseudo labels → supervises Network B
    - Network B produces pseudo labels → supervises Network A
    
    For regression: pseudo labels are the raw predictions (not argmax).
    """
    def __init__(
        self,
        supervised_criterion,
        cps_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        lambda_cps=1.0,
        patience=20,
        min_delta=0.001,
    ):
        """
        Args:
            supervised_criterion: Loss for labeled data (e.g., MSELoss)
            cps_criterion: Loss for CPS pseudo-label supervision (e.g., MSELoss)
            optimizer: Optimizer factory (partial)
            scheduler: Scheduler factory (partial)
            device: Device to run on
            models: List of exactly 2 models with same architecture
            logger: Logger instance
            datamodule: Data module with train/val/test loaders
            lambda_cps: Weight for CPS loss (default: 1.0)
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
        """
        assert len(models) == 2, "CPS requires exactly 2 models"
        
        self.device = device
        self.model1, self.model2 = models
        
        # Loss functions
        self.supervised_criterion = supervised_criterion
        self.cps_criterion = cps_criterion
        self.lambda_cps = lambda_cps
        
        # Optimizers (separate for each network)
        all_params = [p for m in models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)
        
        # Dataloaders
        self.train_labeled_loader = datamodule.train_dataloader()
        self.train_unlabeled_loader = datamodule.unsupervised_train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()
        
        # Logging
        self.logger = logger
        
        # Early stopping
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.best_models_state = None

    def validate(self):
        """Validation using ensemble average of both networks."""
        self.model1.eval()
        self.model2.eval()
        
        val_losses = []
        
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # Ensemble prediction
                pred1 = self.model1(x)
                pred2 = self.model2(x)
                avg_pred = (pred1 + pred2) / 2.0
                
                # Reshape targets to match predictions if needed
                if targets.dim() == 1:
                    targets = targets.unsqueeze(1)
                
                val_loss = F.mse_loss(avg_pred, targets)
                val_losses.append(val_loss.item())
        
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def compute_cps_loss(self, pred1, pred2):
        """
        Compute Cross Pseudo Supervision loss.
        
        For regression:
        - Each network's prediction serves as pseudo label for the other
        - Use L1 or L2 distance between predictions
        
        Args:
            pred1: Predictions from model 1
            pred2: Predictions from model 2
            
        Returns:
            CPS loss (scalar)
        """
        # Model 1 pseudo-labels supervise Model 2
        # Model 2 pseudo-labels supervise Model 1
        # Use detach to stop gradients flowing through pseudo labels
        loss_1_to_2 = self.cps_criterion(pred1, pred2.detach())
        loss_2_to_1 = self.cps_criterion(pred2, pred1.detach())
        
        return loss_1_to_2 + loss_2_to_1

    def train(self, total_epochs, validation_interval):
        """Main training loop with CPS."""
        
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            self.model1.train()
            self.model2.train()
            
            # Metrics tracking
            supervised_losses_log = []
            cps_labeled_losses_log = []
            cps_unlabeled_losses_log = []
            total_losses_log = []
            
            # Create unlabeled iterator (will cycle through it)
            unlabeled_iter = iter(self.train_unlabeled_loader)
            
            # Iterate based on LABELED data (like original trainer)
            # Sample unlabeled batches as needed
            for x_labeled, y_labeled in self.train_labeled_loader:
                self.optimizer.zero_grad()
                
                total_loss = 0.0
                supervised_loss = 0.0
                cps_labeled_loss = 0.0
                cps_unlabeled_loss = 0.0
                
                # ===== LABELED DATA =====
                x_labeled = x_labeled.to(self.device)
                y_labeled = y_labeled.to(self.device)
                
                # Reshape targets if needed
                if y_labeled.dim() == 1:
                    y_labeled = y_labeled.unsqueeze(1)
                
                # Forward pass both networks on labeled data
                pred1_labeled = self.model1(x_labeled)
                pred2_labeled = self.model2(x_labeled)
                
                # Supervised loss (both networks)
                supervised_loss = (
                    self.supervised_criterion(pred1_labeled, y_labeled) +
                    self.supervised_criterion(pred2_labeled, y_labeled)
                )
                total_loss += supervised_loss
                
                # CPS loss on labeled data (optional but helps)
                cps_labeled_loss = self.compute_cps_loss(pred1_labeled, pred2_labeled)
                total_loss += self.lambda_cps * cps_labeled_loss
                
                # ===== UNLABELED DATA =====
                # Get one batch of unlabeled data
                try:
                    x_unlabeled, _ = next(unlabeled_iter)
                except StopIteration:
                    # Restart unlabeled iterator if exhausted
                    unlabeled_iter = iter(self.train_unlabeled_loader)
                    x_unlabeled, _ = next(unlabeled_iter)
                
                x_unlabeled = x_unlabeled.to(self.device)
                
                # Forward pass both networks on unlabeled data
                pred1_unlabeled = self.model1(x_unlabeled)
                pred2_unlabeled = self.model2(x_unlabeled)
                
                # CPS loss on unlabeled data (main semi-supervised signal)
                cps_unlabeled_loss = self.compute_cps_loss(pred1_unlabeled, pred2_unlabeled)
                total_loss += self.lambda_cps * cps_unlabeled_loss
                
                # Backward pass
                total_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model1.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(self.model2.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                # Log metrics
                supervised_losses_log.append(supervised_loss.item())
                cps_labeled_losses_log.append(cps_labeled_loss.item())
                cps_unlabeled_losses_log.append(cps_unlabeled_loss.item())
                total_losses_log.append(total_loss.item())
            
            # Step scheduler
            self.scheduler.step()
            
            # Compute average losses
            summary_dict = {
                "supervised_loss": np.mean(supervised_losses_log),
                "cps_labeled_loss": np.mean(cps_labeled_losses_log),
                "cps_unlabeled_loss": np.mean(cps_unlabeled_losses_log),
                "total_loss": np.mean(total_losses_log),
                "learning_rate": self.optimizer.param_groups[0]['lr'],
            }
            
            # Validation
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
                
                # Early stopping check
                current_val_loss = val_metrics["val_MSE"]
                if current_val_loss < self.best_val_loss - self.min_delta:
                    # Improvement detected
                    self.best_val_loss = current_val_loss
                    self.patience_counter = 0
                    # Save best model states
                    self.best_models_state = [
                        self.model1.state_dict(),
                        self.model2.state_dict()
                    ]
                    summary_dict["early_stop_best"] = True
                else:
                    # No improvement
                    self.patience_counter += 1
                    summary_dict["early_stop_patience"] = self.patience_counter
                
                # Check if patience exceeded
                if self.patience_counter >= self.patience:
                    print(f"\nEarly stopping triggered at epoch {epoch}. Best val_MSE: {self.best_val_loss:.4f}")
                    # Restore best model weights
                    if self.best_models_state is not None:
                        self.model1.load_state_dict(self.best_models_state[0])
                        self.model2.load_state_dict(self.best_models_state[1])
                        print("Restored best model weights")
                    break
            
            self.logger.log_dict(summary_dict, step=epoch)


class NCrossPseudoSupervision:
    """n-Cross Pseudo-Supervision trainer"""

    def __init__(
        self,
        supervised_criterion,
        cps_loss_weight,
        optimizer,
        scheduler,
        device,
        models,
        datamodule,
        logger,
    ):
        self.supervised_criterion = supervised_criterion
        self.cps_loss_weight = cps_loss_weight
    
        # Optimizers (separate for each network)
        all_params = [p for m in models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)
        
        self.device = device
        self.models = models
        if len(models) < 2:
            raise ValueError(
                f"At least two models are required. Models supplied: {len(models)}"
            )

        # Dataloader setup
        self.train_labeled_dataloader = datamodule.train_dataloader()
        self.train_unlabeled_dataloader = datamodule.unsupervised_train_dataloader()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

    def validate(self):
        for model in self.models:
            model.eval()

        val_losses = []

        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                # Ensemble prediction
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)

                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())
        val_loss = np.mean(val_losses)
        return {"val_MSE": val_loss}

    def train(self, total_epochs: int, validation_interval: int) -> None:
        for epoch in (progress_bar := tqdm(range(1, total_epochs + 1))):
            for model in self.models:
                model.train()

            total_losses_logged = []
            supervised_losses_logged = []
            cps_losses_logged = []

            for (x, targets), (x_unlabeled, _) in zip(
                self.train_labeled_dataloader, self.train_unlabeled_dataloader
            ):
                x = x.to(self.device)
                targets = targets.to(self.device)
                x_unlabeled = x_unlabeled.to(self.device)

                self.optimizer.zero_grad()

                # Forward passes for each model with gradient to use as predictions
                preds_labeled = [model(x) for model in self.models]
                preds_unlabeled = [model(x_unlabeled) for model in self.models]
                # Forward passes without gradient to use as targets
                with torch.no_grad():
                    preds_labeled_no_grad = [model(x) for model in self.models]
                    preds_unlabeled_no_grad = [
                        model(x_unlabeled) for model in self.models
                    ]

                # Supervised loss
                supervised_loss_per_model = [
                    self.supervised_criterion(pred, targets) for pred in preds_labeled
                ]
                supervised_loss = sum(supervised_loss_per_model)
                supervised_losses_logged.append(
                    supervised_loss.detach().item() / len(self.models)  # type: ignore
                )

                # CPS loss
                cps_labeled_loss_per_model = []
                cps_unlabeled_loss_per_model = []
                for i in range(len(self.models)):
                    model_cps_labeled_loss = torch.stack(
                        [
                            self.supervised_criterion(
                                preds_labeled[i], preds_labeled_no_grad[j]
                            )
                            for j in set(range(len(self.models))) - {i}
                        ]
                    ).mean()
                    model_cps_unlabeled_loss = torch.stack(
                        [
                            self.supervised_criterion(
                                preds_unlabeled[i], preds_unlabeled_no_grad[j]
                            )
                            for j in set(range(len(self.models))) - {i}
                        ]
                    ).mean()
                    cps_labeled_loss_per_model.append(model_cps_labeled_loss)
                    cps_unlabeled_loss_per_model.append(model_cps_unlabeled_loss)
                cps_labeled_loss = torch.stack(cps_labeled_loss_per_model).mean()
                cps_unlabeled_loss = torch.stack(cps_unlabeled_loss_per_model).mean()
                cps_loss = cps_labeled_loss + cps_unlabeled_loss
                cps_losses_logged.append(cps_loss.detach().item())  # type: ignore

                # Total loss
                loss = supervised_loss + self.cps_loss_weight * cps_loss
                total_losses_logged.append(loss.detach().item())  # type: ignore
                loss.backward()  # type: ignore
                self.optimizer.step()

            self.scheduler.step()
            mean_supervised_loss_logged = np.mean(supervised_losses_logged)
            mean_cps_loss_logged = np.mean(cps_losses_logged)
            mean_total_loss_logged = np.mean(total_losses_logged)

            summary_dict = {
                "supervised_loss": mean_supervised_loss_logged,
                "cps_loss": mean_cps_loss_logged,
                "total_loss": mean_total_loss_logged,
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                progress_bar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)

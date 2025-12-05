import copy
from itertools import cycle

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
        early_stopping=False,
        gradient_clipping=False,
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
        self.early_stopping = early_stopping
        self.gradient_clipping = gradient_clipping
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


    def test(self):
        for model in self.models:
            model.eval()

        test_losses = []
        
        with torch.no_grad():
            for x, targets in self.test_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                
                # Ensemble prediction
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)
                
                test_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                test_losses.append(test_loss.item())
        test_loss = np.mean(test_losses)
        return {"test_MSE": test_loss}


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
                if self.gradient_clipping:
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
                if self.early_stopping and (self.patience_counter >= self.patience):
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
                cps_loss_logged = cps_labeled_loss.detach() + cps_unlabeled_loss.detach()
            
            # Step scheduler
            self.scheduler.step()
            
            # Compute average losses
            summary_dict = {
                "supervised_loss": np.mean(supervised_losses_log),
                "cps_labeled_loss": np.mean(cps_labeled_losses_log),
                "cps_unlabeled_loss": np.mean(cps_unlabeled_losses_log),
                "cps_loss": np.mean(cps_loss_logged),  # type: ignore
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
    """n-Cross Pseudo-Supervision trainer
    
    Note: In this implementaion the CPS loss is additionally scaled by 1/n, therefore
    true CPS loss weight λ is λ = λ'/n, where n is number of models, and λ' is the
    weight `cps_loss_weight` that is supplied to the trainer.
    We are reporting the true λ, not λ'.
    """

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
        if len(optimizer) != 1 and len(models) != len(optimizer):
            raise ValueError(
                f"If multiple optimizers are supplied, the number of optimizers must "
                f"match the number of models. Num of models: {len(models)}, "
                f"num of optimizers: {len(optimizer)}"
            )
        if len(optimizer) != len(scheduler):
            raise ValueError(
                f"Number of optimizers must match the number of schedulers."
                f"Num of optimizers: {len(optimizer)}, "
                f"num of schedulers: {len(scheduler)}"
            )

        if len(optimizer) == 1:
            all_params = [p for m in models for p in m.parameters()]
            print("Using a single optimizer for all models")
            optim = optimizer[0](params=all_params)
            self.optimizers = [optim]
            self.schedulers = [scheduler[0](optimizer=optim)]
        else:
            self.optimizers = []
            self.schedulers = []
            for mod, optim, sched in zip(models, optimizer, scheduler):
                optim = optim(params=mod.parameters())
                print(f"Using optimizer {optim} for model {type(mod).__name__}")
                sched = sched(optimizer=optim)
                self.optimizers.append(optim)
                self.schedulers.append(sched)
        
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


    def test(self):
        for model in self.models:
            model.eval()

        test_losses = []

        with torch.no_grad():
            for x, targets in self.test_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)

                # Use average prediction of all models
                preds = [model(x) for model in self.models]
                avg_preds = torch.stack(preds).mean(0)

                test_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                test_losses.append(test_loss.item())
        test_loss = np.mean(test_losses)
        return {"test_MSE": test_loss}


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

                for optimizer in self.optimizers:
                    optimizer.zero_grad()

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
                    # 1/(n-1) * MSE(y^_i, y^_j)
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
                # This should have been sum, now I get 1/(n-1) and then scale by 1/n again
                cps_labeled_loss = torch.stack(cps_labeled_loss_per_model).mean()
                cps_unlabeled_loss = torch.stack(cps_unlabeled_loss_per_model).mean()
                cps_loss = cps_labeled_loss + cps_unlabeled_loss
                cps_losses_logged.append(cps_loss.detach().item())  # type: ignore

                # Total loss
                loss = supervised_loss + self.cps_loss_weight * cps_loss
                total_losses_logged.append(loss.detach().item())  # type: ignore
                loss.backward()  # type: ignore
                for optimizer in self.optimizers:
                    optimizer.step()

            for scheduler in self.schedulers:
                scheduler.step()
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


class MeanTeacher:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        use_mean_teacher: bool = True,
        ema_init: float = 0.95,      # starting EMA decay
        ema_target: float = 0.99,    # final EMA decay
        ema_ramp_epochs: int = 50,   # epochs to ramp EMA from init->target
        lambda_max: float = 0.01,     # max unsupervised weight
        lambda_ramp_epochs: int = 20,# epochs to ramp lambda_unsup 0->lambda_max
        grad_clip: float = 1.0       # gradient clipping norm (None to disable)
    ):
        self.device = device
        self.models = models

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_params = [p for m in self.models for p in m.parameters()]
        self.optimizer = optimizer(params=all_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloaders (explicit)
        self.supervised_loader = datamodule.train_dataloader()
        self.unsupervised_loader = getattr(datamodule, "unsupervised_train_dataloader", lambda: None)()
        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        # Logging
        self.logger = logger

        # Mean Teacher setup (optional)
        self.use_mean_teacher = use_mean_teacher
        # EMA ramp parameters
        self.ema_init = ema_init
        self.ema_target = ema_target
        self.ema_ramp_epochs = ema_ramp_epochs
        # unsup weight ramp
        self.lambda_max = lambda_max
        self.lambda_ramp_epochs = lambda_ramp_epochs
        # grad clip
        self.grad_clip = grad_clip

        # initialize teachers if needed
        if self.use_mean_teacher:
            self.teacher_models = [copy.deepcopy(m).to(self.device) for m in self.models]
            for t in self.teacher_models:
                t.eval()
                for p in t.parameters():
                    p.requires_grad = False
        else:
            self.teacher_models = None

        # global step counter (for optional fine control)
        self.global_step = 0

    def current_ema_decay(self, epoch):
        """Linearly ramp EMA decay from ema_init -> ema_target over ema_ramp_epochs."""
        if self.ema_ramp_epochs <= 0:
            return self.ema_target
        frac = min(1.0, epoch / float(self.ema_ramp_epochs))
        return float(self.ema_init + frac * (self.ema_target - self.ema_init))

    def current_lambda(self, epoch):
        """Linearly ramp lambda_unsup from 0 -> lambda_max over lambda_ramp_epochs."""
        if self.lambda_ramp_epochs <= 0:
            return float(self.lambda_max)
        frac = min(1.0, epoch / float(self.lambda_ramp_epochs))
        return float(self.lambda_max * frac)

    def update_teacher(self, current_decay):
        """EMA update for each teacher from its student using given decay."""
        if not self.use_mean_teacher:
            return
        for student, teacher in zip(self.models, self.teacher_models):
            for p_s, p_t in zip(student.parameters(), teacher.parameters()):
                p_t.data.mul_(current_decay).add_(p_s.data * (1.0 - current_decay))


    def validate(self):
        models_to_eval = self.teacher_models if self.use_mean_teacher else self.models

        for model in models_to_eval:
            model.eval()

        val_losses = []
        with torch.no_grad():
            for x, targets in self.val_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                preds = [model(x) for model in models_to_eval]
                avg_preds = torch.stack(preds).mean(0)
                val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                val_losses.append(val_loss.item())

        return {"val_MSE": float(np.mean(val_losses))}
    
    def test(self):
        models_to_eval = self.teacher_models if self.use_mean_teacher else self.models

        for model in models_to_eval:
            model.eval()

        test_losses = []
        with torch.no_grad():
            for x, targets in self.test_dataloader:
                x, targets = x.to(self.device), targets.to(self.device)
                preds = [model(x) for model in models_to_eval]
                avg_preds = torch.stack(preds).mean(0)
                test_loss = torch.nn.functional.mse_loss(avg_preds, targets)
                test_losses.append(test_loss.item())

        return {"test_MSE": float(np.mean(test_losses))}

    # def validate(self):
    #     for model in self.models:
    #         model.eval()

    #     val_losses = []
    #     with torch.no_grad():
    #         for x, targets in self.val_dataloader:
    #             x, targets = x.to(self.device), targets.to(self.device)
    #             preds = [model(x) for model in self.models]
    #             avg_preds = torch.stack(preds).mean(0)
    #             val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
    #             val_losses.append(val_loss.item())
    #     return {"val_MSE": float(np.mean(val_losses))}

    def train(self, total_epochs, validation_interval, lambda_unsup=None):
        """
        Paired-batch training with Mean Teacher stability improvements:
        - linear ramp for lambda_unsup
        - linear ramp for EMA decay (teacher)
        - gradient clipping
        - single combined update per paired iteration
        """
        # default lambda_unsup driven by ramp settings unless provided
        if lambda_unsup is None:
            lambda_unsup = self.lambda_max

        sup_batches = len(self.supervised_loader)
        unsup_exists = (self.unsupervised_loader is not None)
        unsup_batches = len(self.unsupervised_loader) if unsup_exists else 0
        print(f"Supervised batches: {sup_batches}, Unsupervised batches: {unsup_batches}")
        print(next(self.teacher_models[0].parameters()).abs().mean())
        print(next(self.models[0].parameters()).abs().mean())


        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            # compute current ramped values
            current_lambda = self.current_lambda(epoch)
            ema_decay_now = self.current_ema_decay(epoch)

            # set models
            for model in self.models:
                model.train()
            if self.teacher_models:
                for t in self.teacher_models:
                    t.eval()

            supervised_losses_logged = []
            unsupervised_losses_logged = []

            # cycle unsup loader so we always have one per sup batch
            if unsup_exists and current_lambda > 0:
                unsup_iter = cycle(self.unsupervised_loader)
            else:
                unsup_iter = None

            for x_sup, y_sup in self.supervised_loader:
                x_sup, y_sup = x_sup.to(self.device), y_sup.to(self.device)

                if unsup_iter is not None:
                    x_unsup, _ = next(unsup_iter)
                    x_unsup = x_unsup.to(self.device)
                else:
                    x_unsup = None

                # supervised loss
                supervised_losses = [self.supervised_criterion(m(x_sup), y_sup) for m in self.models]
                supervised_loss = torch.stack(supervised_losses).mean()

                # unsupervised loss (teacher or student ensemble)
                if (x_unsup is not None) and (current_lambda > 0):
                    with torch.no_grad():
                        if self.use_mean_teacher:
                            teacher_preds = [t(x_unsup) for t in self.teacher_models]
                            pseudo = torch.stack(teacher_preds).mean(0)
                        else:
                            student_preds = [m(x_unsup) for m in self.models]
                            pseudo = torch.stack(student_preds).mean(0)
                    unsup_losses = [torch.nn.functional.mse_loss(m(x_unsup), pseudo) for m in self.models]
                    unsup_loss = torch.stack(unsup_losses).mean()
                else:
                    unsup_loss = None

                # combined loss and step
                total_loss = supervised_loss if unsup_loss is None else supervised_loss + current_lambda * unsup_loss

                self.optimizer.zero_grad()
                total_loss.backward()

                # gradient clipping 
                if (self.grad_clip is not None) and (self.grad_clip > 0):
                    torch.nn.utils.clip_grad_norm_( [p for m in self.models for p in m.parameters() if p.grad is not None], self.grad_clip)

                self.optimizer.step()
                self.global_step += 1

                # update EMA teacher with the current decay
                if self.use_mean_teacher:
                    self.update_teacher(ema_decay_now)

                supervised_losses_logged.append(supervised_loss.item())
                if unsup_loss is not None:
                    unsupervised_losses_logged.append(unsup_loss.item())

            # scheduler step per epoch
            self.scheduler.step()

            # logging & validation
            summary_dict = {
                # "epoch": epoch,
                "supervised_loss": float(np.mean(supervised_losses_logged)) if supervised_losses_logged else 0.0,
                "unsupervised_loss": float(np.mean(unsupervised_losses_logged)) if unsupervised_losses_logged else 0.0,
                "lambda_unsup": float(current_lambda),
                "ema_decay": float(ema_decay_now)
            }

            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)

            self.logger.log_dict(summary_dict, step=epoch)

# # --------------------------------------
# class SemiSupervisedEnsemble:
#     def __init__(
#         self,
#         supervised_criterion,
#         optimizer,
#         scheduler,
#         device,
#         models,                     # list of n student models
#         logger,
#         datamodule,
#         use_mean_teacher: bool = True,
#         ema_init: float = 0.95,
#         ema_target: float = 0.99,
#         ema_ramp_epochs: int = 50,
#         lambda_max: float = 0.01,
#         lambda_ramp_epochs: int = 20,
#         grad_clip: float = 1.0
#     ):
#         self.device = device
#         self.models = models  # n students

#         # Optimizer over all students
#         all_params = [p for m in self.models for p in m.parameters()]
#         self.optimizer = optimizer(params=all_params)
#         self.scheduler = scheduler(optimizer=self.optimizer)
#         self.supervised_criterion = supervised_criterion

#         # Data
#         self.supervised_loader = datamodule.train_dataloader()
#         self.unsupervised_loader = getattr(datamodule, "unsupervised_train_dataloader", lambda: None)()
#         self.val_dataloader = datamodule.val_dataloader()
#         self.test_dataloader = datamodule.test_dataloader()

#         # Logging
#         self.logger = logger

#         # Mean teacher
#         self.use_mean_teacher = use_mean_teacher
#         self.ema_init = ema_init
#         self.ema_target = ema_target
#         self.ema_ramp_epochs = ema_ramp_epochs
#         self.lambda_max = lambda_max
#         self.lambda_ramp_epochs = lambda_ramp_epochs
#         self.grad_clip = grad_clip

#         if self.use_mean_teacher:
#             # single teacher, copied from the first student
#             self.teacher_model = copy.deepcopy(self.models[0]).to(self.device)
#             self.teacher_model.eval()
#             for p in self.teacher_model.parameters():
#                 p.requires_grad = False
#         else:
#             self.teacher_model = None

#         self.global_step = 0

#     def current_ema_decay(self, epoch):
#         frac = min(1.0, epoch / max(1, self.ema_ramp_epochs))
#         return float(self.ema_init + frac * (self.ema_target - self.ema_init))

#     def current_lambda(self, epoch):
#         frac = min(1.0, epoch / max(1, self.lambda_ramp_epochs))
#         return float(self.lambda_max * frac)

#     def update_teacher(self, current_decay):
#         """EMA from the average of all student parameters to single teacher."""
#         if not self.use_mean_teacher:
#             return
#         for teacher_p, *student_params in zip(
#             self.teacher_model.parameters(), *[m.parameters() for m in self.models]
#         ):
#             # average student parameters along all n students
#             avg_student = torch.stack([p.data for p in student_params]).mean(0)
#             teacher_p.data.mul_(current_decay).add_(avg_student * (1 - current_decay))

#     def validate(self):
#         models_to_eval = [self.teacher_model] if self.use_mean_teacher else self.models
#         for m in models_to_eval:
#             m.eval()

#         val_losses = []
#         with torch.no_grad():
#             for x, targets in self.val_dataloader:
#                 x, targets = x.to(self.device), targets.to(self.device)
#                 preds = [m(x) for m in models_to_eval]
#                 avg_preds = torch.stack(preds).mean(0)
#                 val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
#                 val_losses.append(val_loss.item())
#         return {"val_MSE": float(np.mean(val_losses))}

#     def train(self, total_epochs, validation_interval):
#         for epoch in range(1, total_epochs + 1):
#             current_lambda = self.current_lambda(epoch)
#             ema_decay_now = self.current_ema_decay(epoch)

#             for m in self.models:
#                 m.train()
#             if self.teacher_model:
#                 self.teacher_model.eval()

#             if self.unsupervised_loader is not None and current_lambda > 0:
#                 unsup_iter = cycle(self.unsupervised_loader)
#             else:
#                 unsup_iter = None

#             for x_sup, y_sup in self.supervised_loader:
#                 x_sup, y_sup = x_sup.to(self.device), y_sup.to(self.device)
#                 x_unsup = next(unsup_iter)[0].to(self.device) if unsup_iter else None

#                 # Supervised loss
#                 sup_losses = [self.supervised_criterion(m(x_sup), y_sup) for m in self.models]
#                 sup_loss = torch.stack(sup_losses).mean()

#                 # Unsupervised loss
#                 if x_unsup is not None and current_lambda > 0:
#                     with torch.no_grad():
#                         pseudo = self.teacher_model(x_unsup)
#                     unsup_losses = [torch.nn.functional.mse_loss(m(x_unsup), pseudo) for m in self.models]
#                     unsup_loss = torch.stack(unsup_losses).mean()
#                 else:
#                     unsup_loss = None

#                 # Combined loss
#                 total_loss = sup_loss if unsup_loss is None else sup_loss + current_lambda * unsup_loss

#                 self.optimizer.zero_grad()
#                 total_loss.backward()
#                 if self.grad_clip is not None:
#                     torch.nn.utils.clip_grad_norm_(
#                         [p for m in self.models for p in m.parameters() if p.grad is not None],
#                         self.grad_clip
#                     )
#                 self.optimizer.step()
#                 self.global_step += 1

#                 # EMA update
#                 if self.use_mean_teacher:
#                     self.update_teacher(ema_decay_now)

#             if epoch % validation_interval == 0 or epoch == total_epochs:
#                 val_metrics = self.validate()
#             self.logger.log_dict({
#                     "epoch": epoch,
#                     "supervised_loss": float(sup_loss),
#                     "unsupervised_loss": float(unsup_loss) if unsup_loss else 0.0,
#                     "lambda_unsup": float(current_lambda),
#                     "ema_decay": float(ema_decay_now),
#                     **val_metrics
#                 }, step=epoch)




# ---------------------------------------
# from functools import partial

# import numpy as np
# import torch
# from tqdm import tqdm
# from itertools import islice, cycle

# class SemiSupervisedEnsemble:
#     def __init__(
#         self,
#         supervised_criterion,
#         optimizer,
#         scheduler,
#         device,
#         models,
#         logger,
#         datamodule,
#     ):
#         self.device = device
#         self.models = models

#         # Optim related things
#         self.supervised_criterion = supervised_criterion
#         all_params = [p for m in self.models for p in m.parameters()]
#         self.optimizer = optimizer(params=all_params)
#         self.scheduler = scheduler(optimizer=self.optimizer)

#         # # Dataloader setup
#         # self.train_dataloader = datamodule.train_dataloader()
#         # self.val_dataloader = datamodule.val_dataloader()
#         # self.test_dataloader = datamodule.test_dataloader()

#         # datamodule.train_dataloader() now returns [supervised_loader, unsupervised_loader, ...]
#         self.supervised_loader = datamodule.train_dataloader()
#         self.unsupervised_loader = datamodule.unsupervised_train_dataloader()
#         self.val_dataloader = datamodule.val_dataloader()
#         self.test_dataloader = datamodule.test_dataloader()


#         # Logging
#         self.logger = logger

#     def validate(self):
#         for model in self.models:
#             model.eval()

#         val_losses = []
        
#         with torch.no_grad():
#             for x, targets in self.val_dataloader:
#                 x, targets = x.to(self.device), targets.to(self.device)
                
#                 # Ensemble prediction
#                 preds = [model(x) for model in self.models]
#                 avg_preds = torch.stack(preds).mean(0)
                
#                 val_loss = torch.nn.functional.mse_loss(avg_preds, targets)
#                 val_losses.append(val_loss.item())
#         val_loss = np.mean(val_losses)
#         return {"val_MSE": val_loss}
    
#     def train(self, total_epochs, validation_interval, lambda_unsup=0.1):
#         for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
#             for model in self.models:
#                 model.train()

#             supervised_losses_logged = []
#             unsupervised_losses_logged = []

#             # -----------------------------
#             # 1. Supervised pass
#             # -----------------------------
#             for x_sup, y_sup in self.supervised_loader:
#                 x_sup, y_sup = x_sup.to(self.device), y_sup.to(self.device)

#                 self.optimizer.zero_grad()

#                 supervised_losses = [
#                     self.supervised_criterion(model(x_sup), y_sup)
#                     for model in self.models
#                 ]
#                 supervised_loss = torch.stack(supervised_losses).mean()
#                 supervised_loss.backward()
#                 self.optimizer.step()

#                 supervised_losses_logged.append(supervised_loss.item())

#             # -----------------------------
#             # 2. Unsupervised pass  (OUTSIDE supervised loop!)
#             # -----------------------------
#             if lambda_unsup > 0:
#                 for x_unsup, _ in self.unsupervised_loader:
#                     x_unsup = x_unsup.to(self.device)

#                     # Create pseudo-label
#                     with torch.no_grad():
#                         pseudo = torch.stack([m(x_unsup) for m in self.models]).mean(0)

#                     # Compute consistency loss
#                     self.optimizer.zero_grad()
#                     losses = [
#                         torch.nn.functional.mse_loss(m(x_unsup), pseudo)
#                         for m in self.models
#                     ]
#                     unsup_loss = torch.stack(losses).mean() * lambda_unsup
#                     unsup_loss.backward()
#                     self.optimizer.step()

#                     unsupervised_losses_logged.append(unsup_loss.item())

#             # Scheduler
#             self.scheduler.step()

#             summary_dict = {
#                 "supervised_loss": np.mean(supervised_losses_logged),
#                 "unsupervised_loss": np.mean(unsupervised_losses_logged)
#                                     if unsupervised_losses_logged else 0.0
#             }

#             if epoch % validation_interval == 0 or epoch == total_epochs:
#                 val_metrics = self.validate()
#                 summary_dict.update(val_metrics)
#                 pbar.set_postfix(summary_dict)

#             self.logger.log_dict(summary_dict, step=epoch)


#     # def train(self, total_epochs, validation_interval):
#     #     #self.logger.log_dict()
#     #     for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
#     #         for model in self.models:
#     #             model.train()
#     #         supervised_losses_logged = []
#     #         for x, targets in self.train_dataloader:
#     #             x, targets = x.to(self.device), targets.to(self.device)
#     #             self.optimizer.zero_grad()
#     #             # Supervised loss
#     #             supervised_losses = [self.supervised_criterion(model(x), targets) for model in self.models]
#     #             supervised_loss = sum(supervised_losses)
#     #             supervised_losses_logged.append(supervised_loss.detach().item() / len(self.models))  # type: ignore
#     #             loss = supervised_loss
#     #             loss.backward()  # type: ignore
#     #             self.optimizer.step()
#     #         self.scheduler.step()
#     #         supervised_losses_logged = np.mean(supervised_losses_logged)

#     #         summary_dict = {
#     #             "supervised_loss": supervised_losses_logged,
#     #         }
#     #         if epoch % validation_interval == 0 or epoch == total_epochs:
#     #             val_metrics = self.validate()
#     #             summary_dict.update(val_metrics)
#     #             pbar.set_postfix(summary_dict)
#     #         self.logger.log_dict(summary_dict, step=epoch)

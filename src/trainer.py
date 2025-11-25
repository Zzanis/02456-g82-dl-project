from functools import partial
from copy import deepcopy
import math

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

class SemiSupervisedMeanTeacher:
    def __init__(
        self,
        supervised_criterion,
        optimizer,
        scheduler,
        device,
        models,
        logger,
        datamodule,
        ema_alpha_rampup: float = 0.9999,
        ema_alpha_final: float = 0.9999,
        consistency_weight: float = 1e-4,
        consistency_rampup_epochs: int | None = 100,
    ):
        self.device = device
        #student model
        self.students = models

        # Optim related things
        self.supervised_criterion = supervised_criterion
        all_student_params = [p for m in self.students for p in m.parameters()]
        self.optimizer = optimizer(params=all_student_params)
        self.scheduler = scheduler(optimizer=self.optimizer)

        # Dataloader setup
        self.train_dataloader = datamodule.train_dataloader()
        self.unlabeled_dataloader = datamodule.unsupervised_train_dataloader()

        self.val_dataloader = datamodule.val_dataloader()
        self.test_dataloader = datamodule.test_dataloader()

        #Ema + Consistency setup
        self.ema_alpha_rampup = ema_alpha_rampup
        self.ema_alpha_final = ema_alpha_final
        self.consistency_weight = consistency_weight
        self.consistency_rampup_epochs = consistency_rampup_epochs

        self.use_ssl = (
            self.consistency_weight is not None
            and self.consistency_weight > 0.0
            and self.unlabeled_dataloader is not None
        )

        if self.use_ssl:
            self.teachers = [deepcopy(m).to(device) for m in self.students]
            for t in self.teachers:
                for p in t.parameters():
                    p.requires_grad = False
        else:
            self.teachers = []


        # Logging
        self.logger = logger

    def _update_teacher_ema(self, epoch: int):
        """EMA-Update teacher parameters with adaptive alpha."""
        if not self.use_ssl:
            return
        
        # Determine the current alpha based on the ramp-up phase
        if epoch <= self.consistency_rampup_epochs:
            current_alpha = self.ema_alpha_rampup
        else:
            current_alpha = self.ema_alpha_final
            
        # Assuming single teacher based on current implementation
        teacher = self.teachers[0] 

        with torch.no_grad():
        # Iterate over corresponding parameters (by index)
            for t_param in teacher.parameters():
                
                # 1. Sum up all student parameters
                student_sum = 0.0
                for student in self.students:
                    s_param = next(p for p in student.parameters() if p.shape == t_param.shape) 
                    student_sum += s_param.data
                
                # 2. Calculate the average student parameter
                student_avg = student_sum / len(self.students)
                
                # 3. Apply the EMA update
                t_param.data.mul_(current_alpha).add_(student_avg * (1 - current_alpha))

    def _get_lambda_t(self, epoch: int, total_epochs: int) -> float:
        """weights for consistency loss ramp-up"""
        if not self.use_ssl:
            return 0.0
        
        if self.consistency_rampup_epochs is None or self.consistency_rampup_epochs <= 0:
            return self.consistency_weight

        t = min(epoch / self.consistency_rampup_epochs, 1.0)
        # classic Gaussian Ramp-up
        ramp = float(1.0 - math.exp(-5.0 * t * t))
        return self.consistency_weight * ramp
    
    def _unpack_batch(self, batch):
        """Allows both (x, y) and PyG Batch with .y."""
        # case: (x, targets)
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, targets = batch
            return x, targets
        # case: PyG Data / Batch
        x = batch
        targets = getattr(batch, "y", None)
        return x, targets
    

    def validate(self):
        
        tmodel = self.teachers
      
        smodel = self.students

        for t in tmodel:
            t.eval()
        
        for s in smodel:
            s.eval()
        val_losses_t = []
        val_losses_s = []

        with torch.no_grad():
            for batch in self.val_dataloader:
                x, targets = self._unpack_batch(batch)
                x, targets = x.to(self.device), targets.to(self.device)

                avg_predsT = [t(x) for t in tmodel]
                avg_predsT = torch.stack(avg_predsT).mean(0)

                val_lossT = torch.nn.functional.mse_loss(avg_predsT, targets)
                val_losses_t.append(val_lossT.item())

                avg_predsS = [s(x) for s in smodel]
                avg_predsS = torch.stack(avg_predsS).mean(0)

                val_lossS = torch.nn.functional.mse_loss(avg_predsS, targets)
                val_losses_s.append(val_lossS.item())


        val_lossT = float(np.mean(val_losses_t)) if val_losses_t else float("nan")
        val_lossS = float(np.mean(val_losses_s)) if val_losses_s else float("nan")

        return {"val_MSE_teacher": val_lossT, "val_MSE_student": val_lossS}

    def train(self, total_epochs, validation_interval):
        #self.logger.log_dict()
        for epoch in (pbar := tqdm(range(1, total_epochs + 1))):
            for student in self.students:
                student.train()
            for teacher in self.teachers:
                teacher.eval()  #no optimization on teachers

            supervised_losses_logged = []
            consistency_losses_logged = []
            lambda_t = self._get_lambda_t(epoch, total_epochs)
            
            if not self.use_ssl:
                for batch in self.train_dataloader:
                    x, targets = self._unpack_batch(batch)
                    x, targets = x.to(self.device), targets.to(self.device)
                    self.optimizer.zero_grad()
                    # Supervised loss
                    supervised_losses = [self.supervised_criterion(student(x), targets) for student in self.students]
                    supervised_loss = sum(supervised_losses) #/ len(self.students)
                    supervised_losses_logged.append(supervised_loss.detach().item() / len(self.students))  # type: ignore
                    loss = supervised_loss
                    loss.backward()  # type: ignore
                    self.optimizer.step()
                    
            else:
                unlabeled_iter = iter(self.unlabeled_dataloader)

                for batch_l in self.train_dataloader:
                    x_l, targets_l = self._unpack_batch(batch_l)
                    x_l, targets_l = x_l.to(self.device), targets_l.to(self.device)

                    # Get unlabeled batch
                    try:
                        batch_u = next(unlabeled_iter)
                    except StopIteration:
                        unlabeled_iter = iter(self.unlabeled_dataloader)
                        batch_u = next(unlabeled_iter)

                    x_u, _ = self._unpack_batch(batch_u)
                    x_u = x_u.to(self.device)

                    self.optimizer.zero_grad()

                    # --- Supervised Loss (labeled) ---
                    sup_losses = [
                        self.supervised_criterion(student(x_l), targets_l)
                        for student in self.students
                    ]
                    sup_loss = sum(sup_losses)/ len(self.students)
                    supervised_losses_logged.append(sup_loss.detach().item())  # type: ignore

                    # --- Consistency Loss (unlabeled) ---
                    with torch.no_grad():
                        teacher_preds = [teacher(x_u) for teacher in self.teachers]

                    #x_u_aug = self._augment_unlabeled(x_u)
                    student_preds_u = [student(x_u) for student in self.students]

                    cons_losses = [
                        torch.nn.functional.mse_loss(sp, tp.detach())
                        for sp, tp in zip(student_preds_u, teacher_preds)
                    ]
                    cons_loss = sum(cons_losses) / len(cons_losses)
                    consistency_losses_logged.append(cons_loss.detach().item())

                

                    # total loss
                    
                    loss = sup_loss + lambda_t * cons_loss
                    loss.backward()
                    self.optimizer.step()

                    # ema update
                    self._update_teacher_ema(epoch)

            self.scheduler.step()
            supervised_mean = float(np.mean(supervised_losses_logged)) if supervised_losses_logged else float("nan")
            consistency_mean = (
                float(np.mean(consistency_losses_logged))
                if consistency_losses_logged
                else 0.0
            )

            summary_dict = {
                "supervised_loss": supervised_mean,
                "consistency_loss": consistency_mean,
                "lambda_t": lambda_t,

            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                pbar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)


class SemiSupervisedEnsemble:
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
            with torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
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
                with torch.cuda.amp.autocast(enabled=self.amp_enabled):
                    supervised_losses = [
                        self.supervised_criterion(model(x), targets) for model in self.models
                        ]
                    supervised_loss = sum(supervised_losses) / len(self.models)

                #supervised_losses = [self.supervised_criterion(model(x), targets) for model in self.models]
                #supervised_loss = sum(supervised_losses)

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

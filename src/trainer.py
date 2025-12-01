# add these imports at top of your trainer file
import copy

import torch
import numpy as np
from tqdm import tqdm
from functools import partial
from itertools import islice, cycle

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

import numpy as np
import torch
from tqdm import tqdm


class NCrossPseudoSupervision:
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
        self.optimizer = optimizer
        self.scheduler = scheduler
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
                "loss": mean_total_loss_logged,
            }
            if epoch % validation_interval == 0 or epoch == total_epochs:
                val_metrics = self.validate()
                summary_dict.update(val_metrics)
                progress_bar.set_postfix(summary_dict)
            self.logger.log_dict(summary_dict, step=epoch)

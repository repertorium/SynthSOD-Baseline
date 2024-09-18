""" Manager for X-UMX.
Taken from the MUSDB18 example of the asteroid library with minor modifications.
"""

import torch
from asteroid.engine.system import System


class XUMXManager(System):
    """A class for X-UMX systems inheriting the base system class of lightning.

    Slightly modified from the original one on the asteroid MUSDB18 example:
        * Avoid problems with the 'targets' field in config.
        * Remove custom validation step.

    Args:
        model (torch.nn.Module): Instance of model.
        optimizer (torch.optim.Optimizer): Instance or list of optimizers.
        loss_func (callable): Loss function with signature
            (est_targets, targets).
        train_loader (torch.utils.data.DataLoader): Training dataloader.
        val_loader (torch.utils.data.DataLoader): Validation dataloader.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Instance, or list
            of learning rate schedulers. Also supports dict or list of dict as
            ``{"interval": "step", "scheduler": sched}`` where ``interval=="step"``
            for step-wise schedulers and ``interval=="epoch"`` for classical ones.
        config: Anything to be saved with the checkpoints during training.
            The config dictionary to re-instantiate the run for example.
        val_dur (float): When calculating validation loss, the loss is calculated
            per this ``val_dur'' in seconds on GPU to prevent memory overflow.

    For more info on its methods, properties and hooks, have a look at lightning's docs:
    https://pytorch-lightning.readthedocs.io/en/stable/lightning_module.html#lightningmodule-api
    """

    default_monitor: str = "val_loss"

    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        train_loader,
        val_loader=None,
        scheduler=None,
        config=None,
        val_dur=None,
    ):
        config["data"].pop("sources")
        config["data"].pop("targets")
        config["data"].pop("source_augmentations")
        super().__init__(model, optimizer, loss_func, train_loader, val_loader, scheduler, config)
        self.val_dur_samples = model.sample_rate * val_dur



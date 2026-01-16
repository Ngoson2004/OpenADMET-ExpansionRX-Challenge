import json
from chemprop import models
from chemprop import nn 
import torch


class MPNNWithWeightDecay(models.MPNN):
    def __init__(self, *args, weight_decay: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_decay = float(weight_decay)

    def configure_optimizers(self):
        """
        Start from Chemprop's default configure_optimizers (Adam + Noam-like scheduler),
        then inject weight_decay into the optimizer param groups.
        """
        cfg = super().configure_optimizers()

        if self.weight_decay <= 0:
            return cfg  # behave like default Chemprop

        # cfg is a dict: {"optimizer": opt, "lr_scheduler": {...}}
        if isinstance(cfg, dict) and "optimizer" in cfg:
            opt = cfg["optimizer"]
        else:
            # fallback in case of future changes
            opt = cfg

        for group in opt.param_groups:
            group["weight_decay"] = self.weight_decay

        return cfg

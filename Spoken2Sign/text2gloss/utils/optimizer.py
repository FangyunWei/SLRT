# coding: utf-8
"""
Collection of builder functions
"""
from typing import Callable, Optional, Generator

import torch
from torch import nn

# Learning Rate Scheduler
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler

# Optimization Algorithms
from torch.optim import Optimizer
import warnings, math
from utils.misc import get_logger


def build_gradient_clipper(config: dict) -> Optional[Callable]:
    """
    Define the function for gradient clipping as specified in configuration.
    If not specified, returns None.
    Current options:
        - "clip_grad_val": clip the gradients if they exceed this value,
            see `torch.nn.utils.clip_grad_value_`
        - "clip_grad_norm": clip the gradients if their norm exceeds this value,
            see `torch.nn.utils.clip_grad_norm_`
    :param config: dictionary with training configurations
    :return: clipping function (in-place) or None if no gradient clipping
    """
    clip_grad_fun = None
    if "clip_grad_val" in config.keys():
        clip_value = config["clip_grad_val"]
        clip_grad_fun = lambda params: nn.utils.clip_grad_value_(
            parameters=params, clip_value=clip_value
        )
    elif "clip_grad_norm" in config.keys():
        max_norm = config["clip_grad_norm"]
        clip_grad_fun = lambda params: nn.utils.clip_grad_norm_(
            parameters=params, max_norm=max_norm
        )

    if "clip_grad_val" in config.keys() and "clip_grad_norm" in config.keys():
        raise ValueError("You can only specify either clip_grad_val or clip_grad_norm.")

    return clip_grad_fun


def build_optimizer(config: dict, model) -> Optimizer:
    logger = get_logger()
    optimizer_name = config.get("optimizer", "adam").lower()
    weight_decay = config.get("weight_decay", 0)
    eps = config.get("eps", 1.0e-8)
    parameters = []
    base_lr = config['learning_rate'].pop('default')
    for n, p in model.named_children():
        lr_ = base_lr
        for m, lr in config['learning_rate'].items():
            if m in n:
                lr_ = lr
        logger.info('learning rate {}={}'.format(n, lr_))
        parameters.append({'params':p.parameters(), 'lr':lr_})
    # Adam based optimizers
    betas = config.get("betas", (0.9, 0.999))
    amsgrad = config.get("amsgrad", False)

    if optimizer_name == "adam":
        return torch.optim.Adam(
            params=parameters,
            lr=base_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
    elif optimizer_name == "adamw":
        return torch.optim.Adam(
            params=parameters,
            lr=base_lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
    elif optimizer_name == "adagrad":
        return torch.optim.Adagrad(
            params=parameters,
            lr=base_lr,
            lr_decay=config.get("lr_decay", 0),
            weight_decay=weight_decay,
            eps=eps,
        )
    elif optimizer_name == "adadelta":
        return torch.optim.Adadelta(
            params=parameters,
            rho=config.get("rho", 0.9),
            eps=eps,
            lr=base_lr,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            params=parameters,
            lr=base_lr,
            momentum=config.get("momentum", 0),
            alpha=config.get("alpha", 0.99),
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            params=parameters,
            lr=base_lr,
            momentum=config.get("momentum", 0),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError("Unknown optimizer {}.".format(optimizer_name))


def build_scheduler(
    config: dict, optimizer: Optimizer, scheduler_mode: str='max', hidden_size: int = 0
) -> (Optional[lr_scheduler._LRScheduler], Optional[str]):
    """
    Create a learning rate scheduler if specified in config and
    determine when a scheduler step should be executed.
    Current options:
        - "plateau": see `torch.optim.lr_scheduler.ReduceLROnPlateau`
        - "decaying": see `torch.optim.lr_scheduler.StepLR`
        - "exponential": see `torch.optim.lr_scheduler.ExponentialLR`
        - "noam": see `joeynmt.transformer.NoamScheduler`
    If no scheduler is specified, returns (None, None) which will result in
    a constant learning rate.
    :param config: training configuration
    :param optimizer: optimizer for the scheduler, determines the set of
        parameters which the scheduler sets the learning rate for
    :param scheduler_mode: "min" or "max", depending on whether the validation
        score should be minimized or maximized.
        Only relevant for "plateau".
    :param hidden_size: encoder hidden size (required for NoamScheduler)
    :return:
        - scheduler: scheduler object,
        - scheduler_step_at: either "validation" or "epoch"
    """
    scheduler_name = config["scheduler"].lower()

    if scheduler_name == "plateau":
        # learning rate scheduler
        return (
            lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=scheduler_mode,
                verbose=False,
                threshold_mode="abs",
                factor=config.get("decrease_factor", 0.1),
                patience=config.get("patience", 10),
            ),
            "validation",
        )
    elif scheduler_name == "cosineannealing":
        return (
            lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                eta_min=config.get("eta_min", 0),
                T_max=config.get("t_max", 20),
            ),
            "epoch",
        )
    elif scheduler_name == 'warmup_cosineannealing':
        return None
    elif scheduler_name == "cosineannealingwarmrestarts":
        return (
            lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=config.get("t_init", 10),
                T_mult=config.get("t_mult", 2),
            ),
            "step",
        )
    elif scheduler_name == "decaying":
        return (
            lr_scheduler.StepLR(
                optimizer=optimizer, step_size=config.get("decaying_step_size", 1)
            ),
            "epoch",
        )
    elif scheduler_name == "multi_step":
        return (
            lr_scheduler.MultiStepLR(
                optimizer=optimizer, milestones=[40,60], gamma=0.5
            ),
            "epoch",
        )
    elif scheduler_name == "exponential":
        return (
            lr_scheduler.ExponentialLR(
                optimizer=optimizer, gamma=config.get("decrease_factor", 0.99)
            ),
            "epoch",
        )
    elif scheduler_name == "noam":
        factor = config.get("learning_rate_factor", 1)
        warmup = config.get("learning_rate_warmup", 4000)
        return (
            NoamScheduler(
                hidden_size=hidden_size,
                factor=factor,
                warmup=warmup,
                optimizer=optimizer,
            ),
            "step",
        )
    elif scheduler_name == "warmupexponentialdecay":
        min_rate = config.get("learning_rate_min", 1.0e-5)
        decay_rate = config.get("learning_rate_decay", 0.1)
        warmup = config.get("learning_rate_warmup", 4000)
        peak_rate = config.get("learning_rate_peak", 1.0e-3)
        decay_length = config.get("learning_rate_decay_length", 10000)
        return (
            WarmupExponentialDecayScheduler(
                min_rate=min_rate,
                decay_rate=decay_rate,
                warmup=warmup,
                optimizer=optimizer,
                peak_rate=peak_rate,
                decay_length=decay_length,
            ),
            "step",
        )
    else:
        raise ValueError("Unknown learning scheduler {}.".format(scheduler_name))


class NoamScheduler:
    """
    The Noam learning rate scheduler used in "Attention is all you need"
    See Eq. 3 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(
        self,
        hidden_size: int,
        optimizer: torch.optim.Optimizer,
        factor: float = 1,
        warmup: int = 4000,
    ):
        """
        Warm-up, followed by learning rate decay.
        :param hidden_size:
        :param optimizer:
        :param factor: decay factor
        :param warmup: number of warmup steps
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.hidden_size = hidden_size
        self._rate = 0

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self._compute_rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate

    def _compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        return self.factor * (
            self.hidden_size ** (-0.5)
            * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    # pylint: disable=no-self-use
    def state_dict(self):
        return None


class WarmupExponentialDecayScheduler:
    """
    A learning rate scheduler similar to Noam, but modified:
    Keep the warm up period but make it so that the decay rate can be tuneable.
    The decay is exponential up to a given minimum rate.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        peak_rate: float = 1.0e-3,
        decay_length: int = 10000,
        warmup: int = 4000,
        decay_rate: float = 0.5,
        min_rate: float = 1.0e-5,
    ):
        """
        Warm-up, followed by exponential learning rate decay.
        :param peak_rate: maximum learning rate at peak after warmup
        :param optimizer:
        :param decay_length: decay length after warmup
        :param decay_rate: decay rate after warmup
        :param warmup: number of warmup steps
        :param min_rate: minimum learning rate
        """
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.decay_length = decay_length
        self.peak_rate = peak_rate
        self._rate = 0
        self.decay_rate = decay_rate
        self.min_rate = min_rate

    def step(self):
        """Update parameters and rate"""
        self._step += 1
        rate = self._compute_rate()
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate

    def _compute_rate(self):
        """Implement `lrate` above"""
        step = self._step
        warmup = self.warmup

        if step < warmup:
            rate = step * self.peak_rate / warmup
        else:
            exponent = (step - warmup) / self.decay_length
            rate = self.peak_rate * (self.decay_rate ** exponent)
        return max(rate, self.min_rate)

    # pylint: disable=no-self-use
    def state_dict(self):
        return None

# class WarmupCosineannealing(LRScheduler):
#     #based on https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#CosineAnnealingLR
#     def __init__(self, optimizer, T_max, warmup=0, eta_min=0, last_epoch=-1, verbose=False):
#         self.T_max = T_max
#         self.warmup = warmup #epoch
#         self.eta_min = eta_min
#         super(WarmupCosineannealing, self).__init__(optimizer, last_epoch, verbose)   

#     def get_lr(self):
#         # add warmup (to-do)
#         if not self._get_lr_called_within_step:
#             warnings.warn("To get the last learning rate computed by the scheduler, "
#                           "please use `get_last_lr()`.", UserWarning)

#         if self.last_epoch == 0:
#             if self.warmup==0: #no warmup
#                 return [group['lr'] for group in self.optimizer.param_groups]
#             else:
#                 return [0 for group in self.optimizer.param_groups]
#         elif self.last_epoch < self.warmup:
#             #warmup

#         elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
#             return [group['lr'] + (base_lr - self.eta_min) *
#                     (1 - math.cos(math.pi / self.T_max)) / 2
#                     for base_lr, group in
#                     zip(self.base_lrs, self.optimizer.param_groups)]
#         return [(1 + math.cos(math.pi * self.last_epoch / self.T_max)) /
#                 (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max)) *
#                 (group['lr'] - self.eta_min) + self.eta_min
#                 for group in self.optimizer.param_groups]

#     def _get_closed_form_lr(self):
#         #add warmup (to-do)
#         return [self.eta_min + (base_lr - self.eta_min) *
#                 (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
#                 for base_lr in self.base_lrs]


class WarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epochs, last_epoch=-1):
        self.total_epochs = total_epochs
        super().__init__(optimizer, last_epoch)
    def get_lr(self):
        #do not start from zero
        if self.last_epoch<=0:
            return [base_lr*1/(self.total_epochs+1e-8) for base_lr in self.base_lrs]
        else:
            return [base_lr*self.last_epoch/(self.total_epochs+1e-8) for base_lr in self.base_lrs]
    
    def finish(self):
        return self.last_epoch>=self.total_epochs
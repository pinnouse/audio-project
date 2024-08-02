import math
import os
import random
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn


def create_model(n_classes: int = 512, device: torch.device = 'cpu'
                 ) -> nn.Module:
    """Creates a new CNN.

    Creates a new pytorch neural network that models the audio cortex
    using the architecture presented by Josh McDermmott. One key difference
    is that batch normalization in this model occurs over the whole minibatch
    instead of the original 5 adjacent, zero-padded convolution window.

    Args:
    ----
        n_classes: The number of classes the model should output to.
        device: The PyTorch hardware device to send the model to.

    Returns:
    -------
        A CNN model following Josh McDermmott's architecture.

    """
    return nn.Sequential(
        nn.Conv2d(1, out_channels=96, kernel_size=9, stride=3, padding=3),
        nn.ReLU(),
        # nn.BatchNorm2d(96), # McDermmott does normalization differently*
        nn.LocalResponseNorm(5, 0.001, 0.75, 1),
        nn.AvgPool2d(3, stride=2),
        nn.Conv2d(96, out_channels=256, kernel_size=5, stride=2, padding=3),
        nn.ReLU(),
        # nn.BatchNorm2d(256),
        nn.LocalResponseNorm(5, 0.001, 0.75, 1),
        nn.AvgPool2d(3, stride=2, padding=1),
        nn.Conv2d(256, out_channels=512, kernel_size=3, stride=1, padding=2),
        nn.Conv2d(512, out_channels=1024, kernel_size=3, stride=1, padding=2),
        nn.Conv2d(1024, out_channels=512, kernel_size=3, stride=1, padding=2),
        nn.AvgPool2d(3, stride=2),
        nn.Flatten(),
        nn.Linear(8*8*512, 4096),
        nn.Dropout(0.5),
        nn.Linear(4096, n_classes),
        # nn.Softmax(dim=-1)
        ).to(device)

def cv_sets(X: torch.Tensor, y: torch.Tensor, k: int = 10,
            random_seed: Optional[int] = None
            ) -> List[Tuple[List[int], List[int]]]:
    """Generate cross-validation datasets.

    Using scikit-learn's [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html)

    Args:
    ----
        X: The data set to split upon.
        y: The target logits that the data matches to stratify on.
        k: Number of folds.
        test_size: A float representing the portion of data to split into test.
        random_seed: An integer for the random seed used to split the set.

    Returns:
    -------
        A list of `k` tuples, each tuple containing lists of train indices and
        test indices.

    """
    # rs = StratifiedShuffleSplit(n_splits=k, random_state=random_seed, test_size=test_size)
    rs = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_seed)
    return rs.split(X, y)


def batcherize(X_set: torch.Tensor, y_set: torch.Tensor, indices: List[int],
               batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Split model into batches given a list of indices.

    Args:
    ----
        X_set: A tensor for the input data into the model.
        y_set: A tensor for the output data that the model learns to fit.
        indices: A list of indices to create minibatches from.
        batch_size: An integer batch size.

    Returns:
    -------
        A list of ceil(len(indices) / batch_size) to enumerate over, where each
        element is a subset of the training input/target, split in the order
        given by the indices argument.

    """
    batches = []
    n = math.ceil(len(indices) / batch_size)
    for i in range(n):
        start_ix = i * batch_size
        idx = indices[start_ix:start_ix + batch_size]
        batches.append(
            (X_set[idx], y_set[idx])
        )
    return batches

def checkpoint_model(path: os.PathLike, model: nn.Module, epoch: int,
                     optim: torch.optim.Optimizer, train_loss: float,
                     valid_loss: float) -> None:
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optim.state_dict(),
        'train_loss': train_loss,
        'valid_loss': valid_loss,
    }, path)

def load_model_checkpoint(path: os.PathLike, model: Optional[nn.Module] = None,
                          optim: Optional[torch.optim.Optimizer] = None,
                          device: Optional[torch.device] = None):
    # checkpoint: dict = torch.load(path, map_location=device)
    checkpoint: dict = torch.load(path)
    if model is not None:
        model.load_state_dict(checkpoint.pop('model_state_dict'))
    if optim is not None:
        optim.load_state_dict(checkpoint.pop('optim_state_dict'))
    return checkpoint

def train(model: nn.Module, epochs: int, X_train: torch.Tensor,
          y_train: torch.Tensor, optim: torch.optim.Optimizer, loss_fn: any,
          device: any, k: int = 8, bs: int = 16, save_ckpt: bool = True,
          log_training: bool = True) -> Tuple[List[float], List[float]]:
    """Train a model given parameters.

    Args:
    ----
        model: A model instance to be trained.
        epochs: The number of epochs to train for.
        X_train: A pytorch tensor for the data to train on as input to the
            model.
        y_train: A pytorch tensor representing logits (indices) that the model
            should output, corresponding to the input X_train.
        optim: A pytorch optimizer for stepping the gradients.
        loss_fn: The loss function to which the model is optimizing for,
            typically CrossEntropyLoss for classification.
        device: A pytorch device to load data onto.
        k: An integer number of folds for cross-validation training.
        bs: The batch size for parallelization of training.
        test_size: A decimal value for train/valid split used for the cross
            validation splitting.
        save_ckpt: A boolean indicating whether to save checkpoints of the
            model during training.
        log_training: A boolean for logging training information to stdout.

    Returns:
    -------
        A tuple containing two lists, each the training and validation loss
        respectively, over all epochs of training.

    """
    N = X_train.shape[0]
    N_train = math.ceil((N * (k - 1) / k) / bs)
    N_test = math.ceil((N * 1 / k) / bs)
    start_dt = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    ckpt_dir = os.path.join(os.getcwd(), 'checkpoints', start_dt)
    if save_ckpt:
        os.makedirs(ckpt_dir, exist_ok=True)
    if log_training:
        print('Starting training.')
        print(f'Number of minibatches for training/test: {N_train}/{N_test}')
    t_losses = []
    v_losses = []
    for e in range(epochs):
        if log_training:
            print(f'Starting epoch {e+1} of {epochs}')
        cv_splits = cv_sets(X_train.cpu(), y_train.cpu(), k)

        starttime = datetime.now()
        batch_tl = [] # batch train losses
        batch_vl = []
        for i, (train_ix, valid_ix) in enumerate(cv_splits):
            random.shuffle(train_ix)
            random.shuffle(valid_ix)
            train_batches = batcherize(X_train, y_train, train_ix, bs)
            valid_batches = batcherize(X_train, y_train, valid_ix, bs)
            model.train()
            for inputs, targets in train_batches:
                X, t = inputs.to(device), targets.to(device)
                optim.zero_grad()
                out = model(X)
                loss = loss_fn(out, t)
                loss.backward()
                optim.step()
                batch_tl.append(loss.item())
            model.eval()
            for inputs, targets in valid_batches:
                X, t = inputs.to(device), targets.to(device)
                out = model(X)
                loss = loss_fn(out, t)
                batch_vl.append(loss.item())

        t_loss = np.mean(batch_tl)
        v_loss = np.mean(batch_vl)
        t_losses.append(t_loss)
        v_losses.append(v_loss)
        if log_training:
            secs_elapsed = (datetime.now() - starttime).total_seconds()
            print(f'\tTraining took: {secs_elapsed:0.2f}s')
            print(f'\t\twith train loss: {t_loss:0.6f}')
            print(f'\t\twith valid loss: {v_loss:0.6f}')
        if save_ckpt:
            checkpoint_model(os.path.join(ckpt_dir, f'model-ckpt-e{e+1:03}.pt'),
                             model, e, optim, t_loss, v_loss)
    return t_losses, v_losses

def eval_acc(model: nn.Module, X_test: torch.Tensor, y_test: torch.Tensor,
             device: any, bs: int = 32) -> float:
    model.eval()
    N = X_test.shape[0]
    running_score = 0
    with torch.no_grad():
        for i, (X_batch, y_batch) in enumerate(
            batcherize(X_test, y_test, np.arange(N, dtype=int), bs)):
            X, t = X_batch.to(device), y_batch.to(device)
            o = model(X)
            ks = torch.topk(o, 1).indices.reshape(-1)
            running_score += np.sum((ks == t).numpy(force=True))
    return running_score / N

def load_dataset(path: os.PathLike, inputs_file: str = 'inputs.npy',
                 targets_file: str = 'targets.npy'
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    X = np.load(os.path.join(path, inputs_file))
    T = np.load(os.path.join(path, targets_file))

    N = X.shape[0]
    X = X.reshape((N, 1, 256, 256)) # reshape to 1-channel for convolutions

    return torch.tensor(X, dtype=torch.float32), \
        torch.tensor(T, dtype=torch.long)
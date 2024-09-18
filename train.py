""" Script to train X-UMX with SynthSOD or EnsembleSet.
Most of it is taken from the MUSDB18 example of the asteroid library with some modifications.
"""

import os
import argparse
import json
import yaml
import random

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from asteroid.engine.optimizers import make_optimizer
from asteroid.models import XUMX
from asteroid.utils.parser_utils import parse_args_as_dict

from losses_and_metrics import MultiDomainLoss
from managers import XUMXManager
from datasets import load_synthsod_datasets, load_ensembleset_datasets
from utils import bandwidth_to_max_bin, get_statistics, load_model, prepare_parser_from_dict

from pathlib import Path


# Keys which are not in the conf.yml file can be added here.
# In the hierarchical dictionary created when parsing, the key `key` can be
# found at dic['main_args'][key]

# By default train.py will use all available GPUs.
parser = argparse.ArgumentParser()


def main(conf, args):
    # Set seed for random
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # create output dir if not exist
    exp_dir = Path(args.output)
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Load Datasets
    if args.train_on == "synthsod":
        train_dataset, valid_dataset = load_synthsod_datasets(parser, args)
    elif args.train_on == "ensembleset":
        train_dataset, valid_dataset = load_ensembleset_datasets(parser, args)
    else:
        raise ValueError(f"Unknown dataset: {args.train_on}, only 'synthsod' and "
                         f"'ensembleset' are supported.")

    dataloader_kwargs = (
        {"num_workers": args.num_workers, "pin_memory": True} if torch.cuda.is_available() else {}
    )
    train_sampler = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs
    )
    valid_sampler = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, **dataloader_kwargs)

    # Define model
    if args.pretrained is not None:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics(args, train_dataset)

    max_bin = bandwidth_to_max_bin(train_dataset.sample_rate, args.in_chan, args.bandwidth)

    if args.pretrained_separator is not None:
        x_unmix = load_model(os.path.abspath(args.pretrained_separator))
        x_unmix._return_time_signals = True
        x_unmix.train()
    else:
        x_unmix = XUMX(
            window_length=args.window_length,
            input_mean=scaler_mean,
            input_scale=scaler_std,
            nb_channels=args.nb_channels,
            hidden_size=args.hidden_size,
            in_chan=args.in_chan,
            n_hop=args.nhop,
            sources=args.targets,
            max_bin=max_bin,
            bidirectional=args.bidirectional,
            sample_rate=train_dataset.sample_rate,
            spec_power=args.spec_power,
            return_time_signals=True if args.loss_use_multidomain else False,
        )

    # Define optimizer
    optimizer = make_optimizer(
        x_unmix.parameters(),
        lr=args.lr, optimizer="adam", weight_decay=args.weight_decay
    )

    # Define scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=args.lr_decay_gamma, patience=args.lr_decay_patience, cooldown=10
    )

    # Save config
    conf_path = os.path.join(exp_dir, "conf.yml")
    with open(conf_path, "w") as outfile:
        yaml.safe_dump(conf, outfile)

    # Define Loss function.
    loss_func = MultiDomainLoss(
        window_length=args.window_length,
        in_chan=args.in_chan,
        n_hop=args.nhop,
        spec_power=args.spec_power,
        nb_channels=args.nb_channels,
        loss_combine_sources=args.loss_combine_sources,
        loss_use_multidomain=args.loss_use_multidomain,
        mix_coef=args.mix_coef,
    )

    # Define manager
    system = XUMXManager(
        model=x_unmix,
        loss_func= loss_func,
        optimizer=optimizer,
        train_loader=train_sampler,
        val_loader=valid_sampler,
        scheduler=scheduler,
        config=conf,
        val_dur=args.val_dur,
    )

    # Define callbacks
    callbacks = []
    checkpoint_dir = os.path.join(exp_dir, "checkpoints/")
    monitor, es_mode = 'val_loss', 'min'
    checkpoint = ModelCheckpoint(
        checkpoint_dir, monitor=monitor, mode=es_mode, save_top_k=5, verbose=True
    )
    callbacks.append(checkpoint)

    es = EarlyStopping(monitor=monitor, mode=es_mode, patience=args.patience, verbose=True)
    callbacks.append(es)

    # Define trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=callbacks,
        default_root_dir=exp_dir,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        strategy="ddp_find_unused_parameters_true", #"ddp",
        devices="auto",
        # limit_train_batches=0.1,  # Useful for fast experiment
        # limit_val_batches=0.1,  # Useful for fast experiment
    )

    # Train
    trainer.fit(system)

    # Save best model
    best_k = {k: v.item() for k, v in checkpoint.best_k_models.items()}
    with open(os.path.join(exp_dir, "best_k_models.json"), "w") as f:
        json.dump(best_k, f, indent=0)

    state_dict = torch.load(checkpoint.best_model_path)
    system.load_state_dict(state_dict=state_dict["state_dict"])
    system.cpu()

    to_save = system.model.serialize()
    # to_save.update(train_dataset.get_infos())
    torch.save(to_save, os.path.join(exp_dir, "best_model.pth"))


if __name__ == "__main__":
    # We start with opening the config file conf.yml as a dictionary from
    # which we can create parsers. Each top level key in the dictionary defined
    # by the YAML file creates a group in the parser.
    with open("conf.yml") as f:
        def_conf = yaml.safe_load(f)
    parser = prepare_parser_from_dict(def_conf, parser=parser)

    # Arguments are then parsed into a hierarchical dictionary (instead of
    # flat, as returned by argparse) to facilitate calls to the different
    # asteroid methods (see in main).
    # plain_args is the direct output of parser.parse_args() and contains all
    # the attributes in an non-hierarchical structure. It can be useful to also
    # have it so we included it here.
    arg_dic, plain_args = parse_args_as_dict(parser, return_plain_args=True)
    main(arg_dic, plain_args)

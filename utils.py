""" Miscellaneous functions for training X-UMX.
Most of them are taken from the MUSDB18 example of the asteroid library.
"""

import numpy as np
import torch
import sklearn.preprocessing
import copy
import sys
import tqdm

from asteroid.models import XUMX
from asteroid.models.x_umx import _STFT, _Spectrogram

import argparse
from asteroid.utils.parser_utils import str2bool, str2bool_arg, str_int_float


def load_model(model_name, device="cpu"):
    # Literally taken from the original one on the asteroid MUSDB18 example.
    print("Loading model from: {}".format(model_name), file=sys.stderr)
    conf = torch.load(model_name, map_location="cpu")
    model = XUMX.from_pretrained(model_name)
    model.eval()
    model.to(device)
    return model


def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    # Literally taken from the original one on the asteroid MUSDB18 example.
    freqs = np.linspace(0, float(rate) / 2, n_fft // 2 + 1, endpoint=True)

    return np.max(np.where(freqs <= bandwidth)[0]) + 1


def get_statistics(args, dataset):
    # Literally taken from the original one on the asteroid MUSDB18 example.
    scaler = sklearn.preprocessing.StandardScaler()

    spec = torch.nn.Sequential(
        _STFT(window_length=args.window_length, n_fft=args.in_chan, n_hop=args.nhop),
        _Spectrogram(spec_power=args.spec_power, mono=True),
    )

    dataset_scaler = copy.deepcopy(dataset)
    dataset_scaler.samples_per_track = 1
    dataset_scaler.random_segments = False
    dataset_scaler.random_track_mix = False
    dataset_scaler.segment = False
    pbar = tqdm.tqdm(range(len(dataset_scaler)))
    for ind in pbar:
        x, _ = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        X = spec(torch.from_numpy(x[None, ...]))[0]
        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))
    return scaler.mean_, std


def prepare_parser_from_dict(dic, parser=None):
    """Prepare an argparser from a dictionary.

    Slightly modified from the original asteroid.utils.parser_utils.prepare_parser_from_dict:
        * Allow list of values in the cmd arguments.

    Args:
        dic (dict): Two-level config dictionary with unique bottom-level keys.
        parser (argparse.ArgumentParser, optional): If a parser already
            exists, add the keys from the dictionary on the top of it.

    Returns:
        argparse.ArgumentParser:
            Parser instance with groups corresponding to the first level keys
            and arguments corresponding to the second level keys with default
            values given by the values.
    """

    def standardized_entry_type(value):
        """If the default value is None, replace NoneType by str_int_float.
        If the default value is boolean, look for boolean strings."""
        if value is None:
            return str_int_float
        if isinstance(str2bool(value), bool):
            return str2bool_arg
        return type(value)

    if parser is None:
        parser = argparse.ArgumentParser()
    for k in dic.keys():
        group = parser.add_argument_group(k)
        for kk in dic[k].keys():
            entry_type = standardized_entry_type(dic[k][kk])
            if entry_type == list:
                entry_type = standardized_entry_type(dic[k][kk][0])
                group.add_argument("--" + kk, default=dic[k][kk], type=entry_type, nargs="+")
            else:
                group.add_argument("--" + kk, default=dic[k][kk], type=entry_type)
    return parser

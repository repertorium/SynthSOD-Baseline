""" Loss functions for X-UMX.
Most of them are taken from the MUSDB18 example of the asteroid library with minor modifications.
"""

import itertools
from operator import itemgetter

import torch
from torch import nn
from torch.nn.modules.loss import _Loss

from asteroid.losses import singlesrc_mse
from asteroid.models.x_umx import _STFT, _Spectrogram


def freq_domain_loss(s_hat, gt_spec, nb_channels, combination=True):
    """Calculate frequency-domain loss between estimated and reference spectrograms.
    MSE between target and estimated target spectrograms is adopted as frequency-domain loss.
    If you set ``loss_combine_sources: yes'' in conf.yml, computes loss for all possible
    combinations of 1, ..., nb_sources-1 instruments.

    Slightly modified from the original one on the asteroid MUSDB18 example:
        * Be able to deal with multi-channel audio signals.

    Input:
        estimated spectrograms
            (Sources, Freq. bins, Batch size, Channels, Frames)
        reference spectrograms
            (Freq. bins, Batch size, Sources x Channels, Frames)
        whether use combination or not (optional)
    Output:
        calculated frequency-domain loss
    """

    n_src = len(s_hat)
    idx_list = [i for i in range(n_src)]

    inferences = []
    refrences = []
    for i, s in enumerate(s_hat):
        inferences.append(s)
        refrences.append(gt_spec[..., nb_channels * i : nb_channels * i + nb_channels, :])
    assert inferences[0].shape == refrences[0].shape

    _loss_mse = 0.0
    cnt = 0.0
    for i in range(n_src):
        _loss_mse += singlesrc_mse(inferences[i], refrences[i]).mean()
        cnt += 1.0

    # If Combination is True, calculate the expected combinations.
    if combination:
        for c in range(2, n_src):
            patterns = list(itertools.combinations(idx_list, c))
            for indices in patterns:
                tmp_loss = singlesrc_mse(
                    sum(itemgetter(*indices)(inferences)),
                    sum(itemgetter(*indices)(refrences)),
                ).mean()
                _loss_mse += tmp_loss
                cnt += 1.0

    _loss_mse /= cnt

    return _loss_mse


def time_domain_loss(mix, time_hat, gt_time, combination=True):
    """Calculate weighted time-domain loss between estimated and reference time signals.
    weighted SDR [1] between target and estimated target signals is adopted as time-domain loss.
    If you set ``loss_combine_sources: yes'' in conf.yml, computes loss for all possible
    combinations of 1, ..., nb_sources-1 instruments.

    Slightly modified from the original one on the asteroid MUSDB18 example:
        * Be able to deal with multi-channel audio signals.

    Input:
        mixture time signal
            (Batch size, Channels, Time Length (samples))
        estimated time signals
            (Sources, Batch size, Channels, Time Length (samples))
        reference time signals
            (Batch size, Sources x Channels, Time Length (samples))
        whether use combination or not (optional)
    Output:
        calculated time-domain loss

    References
        - [1] : "Phase-aware Speech Enhancement with Deep Complex U-Net",
          Hyeong-Seok Choi et al. https://arxiv.org/abs/1903.03107
    """

    n_src, n_batch, n_channel, time_length = time_hat.shape
    idx_list = [i for i in range(n_src)]

    # Fix Length
    mix = mix[Ellipsis, :time_length]
    gt_time = gt_time[Ellipsis, :time_length]

    # Prepare Data and Fix Shape
    mix_ref = [mix]
    mix_ref.extend([gt_time[..., n_channel * i : n_channel * i + n_channel, :] for i in range(n_src)])
    mix_ref = torch.stack(mix_ref)
    mix_ref = mix_ref.view(-1, time_length)
    time_hat = time_hat.view(n_batch * n_channel * time_hat.shape[0], time_hat.shape[-1])

    # If Combination is True, calculate the expected combinations.
    if combination:
        indices = []
        for c in range(2, n_src):
            indices.extend(list(itertools.combinations(idx_list, c)))

        for tr in indices:
            sp = [n_batch * n_channel * (tr[i] + 1) for i in range(len(tr))]
            ep = [n_batch * n_channel * (tr[i] + 2) for i in range(len(tr))]
            spi = [n_batch * n_channel * tr[i] for i in range(len(tr))]
            epi = [n_batch * n_channel * (tr[i] + 1) for i in range(len(tr))]

            tmp = sum([mix_ref[sp[i] : ep[i], ...].clone() for i in range(len(tr))])
            tmpi = sum([time_hat[spi[i] : epi[i], ...].clone() for i in range(len(tr))])
            mix_ref = torch.cat([mix_ref, tmp], dim=0)
            time_hat = torch.cat([time_hat, tmpi], dim=0)

        mix_t = mix_ref[: n_batch * n_channel, Ellipsis].repeat(n_src + len(indices), 1)
        refrences_t = mix_ref[n_batch * n_channel :, Ellipsis]
    else:
        mix_t = mix_ref[: n_batch * n_channel, Ellipsis].repeat(n_src, 1)
        refrences_t = mix_ref[n_batch * n_channel :, Ellipsis]

    # Calculation
    _loss_sdr = weighted_sdr(time_hat, refrences_t, mix_t)

    return 1.0 + _loss_sdr


def weighted_sdr(input, gt, mix, weighted=True, eps=1e-10):
    # ``input'', ``gt'' and ``mix'' should be (Batch, Time Length)
    # Literally taken from the original one on the asteroid MUSDB18 example.
    assert input.shape == gt.shape
    assert mix.shape == gt.shape

    ns = mix - gt
    ns_hat = mix - input

    if weighted:
        alpha_num = (gt * gt).sum(1, keepdims=True)
        alpha_denom = (gt * gt).sum(1, keepdims=True) + (ns * ns).sum(1, keepdims=True)
        alpha = alpha_num / (alpha_denom + eps)
    else:
        alpha = 0.5

    # Target
    num_cln = (input * gt).sum(1, keepdims=True)
    denom_cln = torch.sqrt(eps + (input * input).sum(1, keepdims=True)) * torch.sqrt(
        eps + (gt * gt).sum(1, keepdims=True)
    )
    sdr_cln = num_cln / (denom_cln + eps)

    # Noise
    num_noise = (ns * ns_hat).sum(1, keepdims=True)
    denom_noise = torch.sqrt(eps + (ns_hat * ns_hat).sum(1, keepdims=True)) * torch.sqrt(
        eps + (ns * ns).sum(1, keepdims=True)
    )
    sdr_noise = num_noise / (denom_noise + eps)

    return torch.mean(-alpha * sdr_cln - (1.0 - alpha) * sdr_noise)


class MultiDomainLoss(_Loss):
    """A class for calculating loss functions of X-UMX.

    Slightly modified from the original one on the asteroid MUSDB18 example:
        * Be able to deal with multi-channel audio signals.

    Args:
        window_length (int): The length in samples of window function to use in STFT.
        in_chan (int): Number of input channels, should be equal to
            STFT size and STFT window length in samples.
        n_hop (int): STFT hop length in samples.
        spec_power (int): Exponent for spectrogram calculation.
        nb_channels (int): set number of channels for model (1 for mono
            (spectral downmix is applied,) 2 for stereo).
        loss_combine_sources (bool): Set to true if you are using the combination scheme
            proposed in [1]. If you select ``loss_combine_sources: yes'' via
            conf.yml, this is set as True.
        loss_use_multidomain (bool): Set to true if you are using a frequency- and time-domain
            losses collaboratively, i.e., Multi Domain Loss (MDL) proposed in [1].
            If you select ``loss_use_multidomain: yes'' via conf.yml, this is set as True.
        mix_coef (float): A mixing parameter for multi domain losses

    References
        [1] "All for One and One for All: Improving Music Separation by Bridging
        Networks", Ryosuke Sawata, Stefan Uhlich, Shusuke Takahashi and Yuki Mitsufuji.
        https://arxiv.org/abs/2010.04228 (and ICASSP 2021)
    """

    def __init__(
        self,
        window_length,
        in_chan,
        n_hop,
        spec_power,
        nb_channels,
        loss_combine_sources,
        loss_use_multidomain,
        mix_coef,
    ):
        super().__init__()
        self.transform = nn.Sequential(
            _STFT(window_length=window_length, n_fft=in_chan, n_hop=n_hop),
            _Spectrogram(spec_power=spec_power, mono=False),
        )
        self._combi = loss_combine_sources
        self._multi = loss_use_multidomain
        self.coef = mix_coef
        self.nb_channels = nb_channels
        print("Combination Loss: {}".format(self._combi))
        if self._multi:
            print(
                "Multi Domain Loss: {}, scaling parameter for time-domain loss={}".format(
                    self._multi, self.coef
                )
            )
        else:
            print("Multi Domain Loss: {}".format(self._multi))
        self.cnt = 0

    def forward(self, est_targets, targets, return_est=False, **kwargs):
        """est_targets (list) has 2 elements:
            [0]->Estimated Spec. : (Sources, Frames, Batch size, Channels, Freq. bins)
            [1]->Estimated Signal: (Sources, Batch size, Channels, Time Length)

        targets: (Batch, Source, Channels, TimeLen)
        """

        spec_hat = est_targets[0]
        time_hat = est_targets[1]

        # Fix shape and apply transformation of targets
        n_batch, n_src, n_channel, time_length = targets.shape
        targets = targets.view(n_batch, n_src * n_channel, time_length)
        Y = self.transform(targets)[0]

        if self._multi:
            n_src = spec_hat.shape[0]
            mixture_t = sum([targets[:, self.nb_channels * i : self.nb_channels * i + self.nb_channels, ...]
                             for i in range(n_src)])
            loss_f = freq_domain_loss(spec_hat, Y, self.nb_channels, combination=self._combi)
            loss_t = time_domain_loss(mixture_t, time_hat, targets, combination=self._combi)
            loss = float(self.coef) * loss_t + loss_f
        else:
            loss = freq_domain_loss(spec_hat, Y, self.nb_channels, combination=self._combi)

        return loss

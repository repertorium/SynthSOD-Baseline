""" Dataloaders for SynthSOD, EnsembleSet, Aalto Anechoic Orchestral and URMP datasets.
Based on the dataloader for MUSDB18 from the asteroid library.
"""

import warnings

import torch
from pathlib import Path
import numpy as np
from types import SimpleNamespace
import soundfile as sf
import librosa

import random
import tqdm
import json


class SynthSODDataset(torch.utils.data.Dataset):
    """SynthSOD orchestra music separation dataset

    The dataset consists of 467 full lengths music tracks (~50h duration) of
    classical music (orchestras and ensembles) synthesized with the Spitfire's
    BBC Orchestra VST plugin along with their isolated stems:
        `Violin_1`, `Violin_2`, `Viola`, `Cello`, `Bass`,
        `Flute`, `Piccolo`, `Clarinet`, `Oboe`, `coranglais`, `Bassoon`,
        `Horn`, `Trumpet`, `Trombone`, `Tuba`,
        `Harp`, `Timpani`, and `untunedpercussion`.

    This dataset assumes music tracks in (sub)folders where each folder
    might have a different number of sources. Only the sources indicated
    in the 'sources' argument will be loaded and added to the mixture and
    only the sources indicated in the 'targets' argument will be returned
    as targets.

    The option 'random_track_mix' to randomly mix sources from different
    tracks has not been tested and will probably contain bugs.

    Args:
        metadata_file_path (str): Path to the JSON metadata file with the list of
            tracks to load.
        synthsod_data_path (str): Path to the data folder of the SynthSOD dataset
            with the folders of the tracks.
        sources (:obj:`list` of :obj:`str`, optional): List of source names
            to laod to the mixtures. Defaults to the 18 SynthSOD sources.
        targets (list or None, optional): List of source names to be used as targets.
            Defaults to None (all the sources are defined as targets).
        join_violins (bool, optional): Join the Violin_1 and Violin_2 sources into
            one single target. Defaults to True.
        join_piccolo_to_flute (bool, optional): Join the Piccolo source to the Flute
            target. Defaults to True.
        join_coranglais_to_oboe (bool, optional): Join the coranglais source to the
            Oboe target. Defaults to True.
        segment (float, optional): Duration of segments in seconds,
            defaults to ``None`` which loads the full-length audio tracks.
        samples_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to `1`.
        random_segments (boolean, optional): Enables random offset for track segments.
            Default to False.
        fixed_segments (boolean, optional): Always take the same segments in every track.
            Useful for validation. Not compatible with random_track_mix.
            Default to False.
        random_track_mix (boolean, optional): enables mixing of random sources from
            different tracks to assemble mix. Untested, it might contain bugs.
            Default to False.
        convert_to_mono (bool, optional): Convert the audio to mono. Default to True.
        train_minus_one (bool, optional): Return the targets as the minus on of the
            source instead as the separated stem. Default to False.
        source_augmentations (:obj:`list` of :obj:`callable`): list of augmentation
            function names, defaults to no-op augmentations (input = output)
        sample_rate (int, optional): Samplerate of files in dataset.
            It will resample the signals after loading them if they have a different
            sample rate (slow). Default to 44100.
        max_duration (float, optional): Maximum duration of the tracks.
            Tracks longer than max_duration [seconds] will be skipped. Default to None.
        mix_only (bool, optional): Return only the mixture without the targets.
            Default to False.
        fake_musdb_format (bool, optional): Return the data in the format of the MUSDB18
            dataset to be used with the museval library. Default to False.
        size_limit (int/float, optional): Limit the number of tracks to load (if integer)
            or ratio of the dataset (if float between 0 and 1).
            Default to None (load all the tracks).
    """

    dataset_name = "SynthSOD"

    def __init__(
        self,
        metadata_file_path,
        synthsod_data_path,
        sources=('Violin_1', 'Violin_2', 'Viola', 'Cello', 'Bass', 'Flute', 'Piccolo', 'Clarinet', 'Oboe', 'coranglais',
                 'Bassoon', 'Horn', 'Trumpet', 'Trombone', 'Tuba', 'Harp', 'Timpani', 'untunedpercussion'),
        targets=None,
        join_violins=True,
        join_piccolo_to_flute=True,
        join_coranglais_to_oboe=True,
        segment=None,
        samples_per_track=1,
        random_segments=False,
        fixed_segments=False,
        random_track_mix=False,
        convert_to_mono=True,
        train_minus_one=False,
        source_augmentations=lambda audio: audio,
        sample_rate=44100,
        max_duration=None,
        mix_only=False,
        fake_musdb_format=False,
        size_limit=None,
    ):
        assert not (fixed_segments and random_track_mix)
        assert not (mix_only and fake_musdb_format)

        self.metadata_file_path = Path(metadata_file_path).expanduser()
        self.synthsod_data_path = Path(synthsod_data_path).expanduser()

        self.sources = sources
        self.targets = targets if targets is not None else sources
        self.join_violins = join_violins
        self.join_piccolo_to_flute = join_piccolo_to_flute
        self.join_coranglais_to_oboe = join_coranglais_to_oboe
        if join_violins:
            self.targets = [target if target != 'Violin_1' else 'Violin'
                            for target in self.targets if target != 'Violin_2']
        if join_piccolo_to_flute:
            self.targets = [target for target in self.targets if target != 'Piccolo']
        if join_coranglais_to_oboe:
            self.targets = [target for target in self.targets if target != 'coranglais']

        self.segment = int(segment * sample_rate) if segment else None
        self.samples_per_track = samples_per_track
        self.random_segments = random_segments
        self.fixed_segments = fixed_segments
        self.random_track_mix = random_track_mix
        self.convert_to_mono = convert_to_mono
        self.train_minus_one = train_minus_one
        self.source_augmentations = source_augmentations
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.mix_only = mix_only
        self.fake_musdb_format = fake_musdb_format

        self.tracks = self.get_tracks()
        if not self.tracks:
            raise RuntimeError("No tracks found.")
        if fixed_segments:
            self.tracks_start = [None, ] * (len(self.tracks) * self.samples_per_track)
        if size_limit is not None:
            if 0 < size_limit < 1:
                size_limit = int(size_limit * len(self.tracks))
            self.tracks = self.tracks[:size_limit]

    def __getitem__(self, index):
        # load sources
        audio_sources = {}
        for source in self.sources:
            track_id = random.choice(range(len(self.tracks))) if self.random_track_mix \
                else index // self.samples_per_track
            start, end = self.get_track_segment(track_id)
            if source in self.tracks[track_id]:
                try:
                    audio, sr = sf.read(str(self.tracks[track_id][source]), start=start, stop=end, always_2d=True)
                    if sr != self.sample_rate:
                        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=self.sample_rate, res_type='polyphase').T
                except RuntimeError:
                    print(f"Failed to open {Path(self.tracks[track_id][source])} with start={start} and end={end}.")
                    print(f"Replacing the source by silence.")
                    audio = np.zeros((end-start, 2))
                audio = audio.astype(np.float32).T
                if self.convert_to_mono:
                    audio = np.mean(audio, axis=0, keepdims=True)
                audio_sources[source] = self.source_augmentations(audio)
            else:
                audio_sources[source] = np.zeros((1 if self.convert_to_mono else 2, end-start), dtype=np.float32)
        audio_mix = np.sum(list(audio_sources.values()), axis=0)

        if self.mix_only:
            return audio_mix
        else:
            if self.targets:
                if 'Violin' in self.targets and self.join_violins:
                    audio_sources['Violin'] = audio_sources.pop('Violin_1') + audio_sources.pop('Violin_2')
                if 'Flute' in self.targets and self.join_piccolo_to_flute:
                    audio_sources['Flute'] += audio_sources.pop('Piccolo')
                if 'Oboe' in self.targets and self.join_coranglais_to_oboe:
                    audio_sources['Oboe'] += audio_sources.pop('coranglais')
                audio_sources = np.stack([audio_sources[target] for target in self.targets], axis=0)
                if self.train_minus_one:
                    audio_sources = audio_mix - audio_sources
            if self.fake_musdb_format:
                sample_source = list(self.tracks[track_id].keys())[0]  # A source present in the track
                fake_musdb_track = SimpleNamespace()
                fake_musdb_track.name = Path(self.tracks[track_id][sample_source]).parts[-3]
                fake_musdb_track.folder = Path(self.tracks[track_id][sample_source]).parts[-3]
                fake_musdb_track.rate = self.sample_rate
                fake_musdb_track.subset = ""
                fake_musdb_track.audio = audio_mix.T
                fake_musdb_track.targets = {target_name: SimpleNamespace() for target_name in self.targets}
                for target_name, target_audio in zip(self.targets, audio_sources):
                    fake_musdb_track.targets[target_name].audio = target_audio.T
                return fake_musdb_track
            else:
                return audio_mix, audio_sources

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def get_track_segment(self, track_idx):
        """Return the segment of the track to load"""
        track_info = sf.info(list(self.tracks[track_idx].values())[0])  # Get the path of the first source (all sources have the same length)
        track_len = track_info.frames
        if self.segment is not None and self.segment != 0:
            if self.random_segments:
                if self.fixed_segments:
                    if self.tracks_start[track_idx] is None:
                        self.tracks_start[track_idx] = int(random.uniform(0, track_len - self.segment))
                    start = self.tracks_start[track_idx]
                    end = start + self.segment
                else:
                    start = int(random.uniform(0, track_len - self.segment))
                    end = start + self.segment
            else:
                start = 0
                end = track_len
        else:
            start = 0
            end = track_len
        return start, end

    def get_tracks(self):
        """Return the path to the midi files and the information to synthesize them"""
        with open(self.metadata_file_path, 'r') as fp:
            db_info_dict = json.load(fp)

        p = self.synthsod_data_path

        tracks = []
        for song_info in db_info_dict['songs'].values():
            if self.max_duration is None or song_info['duration'] <= self.max_duration:
                track_path = p / song_info['song_name'] / 'Tree'
                if track_path.is_dir():
                    sources_paths = {}
                    for source in self.sources:
                        if (track_path / (source+'.flac')).is_file():
                            sources_paths[source] = str(track_path / (source+'.flac'))
                    # TODO: Check that the track is at least as long as the requested segments?
                    tracks.append(sources_paths)
                else:
                    warnings.warn(f"Track {track_path} not found")

        return tracks


class EnsembleSetDataset(torch.utils.data.Dataset):
    """EnsembleSet separation dataset

    The dataset consists of 80 full lengths music tracks (~6h duration) of
    classical music (ensembles) synthesized with the Spitfire's
    BBC Orchestra VST plugin along with their isolated stems:
        `Violin`, `Viola`, `Cello`, `Bass`,
        `Flute`, `Clarinet`, `Oboe`, `Bassoon`,
        `Horn`, `Trumpet`, and `Timpani`.

    This dataset assumes music tracks in (sub)folders which contain a
    subfolder for every microphone with the audio files of the sources.
    Only the sources indicated in the 'sources' argument will be loaded
    and added to the mixture and  only the sources indicated in the
    'targets' argument will be returned as targets.

    The option 'random_track_mix' to randomly mix sources from different
    tracks has not been tested and will probably contain bugs.

    Args:
        ensembleset_root_path (str): Path to the data folder of the SynthSOD dataset
            with the folders of the tracks.
        sources (:obj:`list` of :obj:`str`, optional): List of source names
            to laod to the mixtures. Defaults to the 18 SynthSOD sources.
        targets (list or None, optional): List of source names to be used as targets.
            Defaults to None (all the sources are defined as targets).
        segment (float, optional): Duration of segments in seconds,
            defaults to ``None`` which loads the full-length audio tracks.
        samples_per_track (int, optional):
            Number of samples yielded from each track, can be used to increase
            dataset size, defaults to `1`.
        random_segments (boolean, optional): Enables random offset for track segments.
            Default to False.
        fixed_segments (boolean, optional): Always take the same segments in every track.
            Useful for validation. Not compatible with random_track_mix.
            Default to False.
        random_track_mix (boolean, optional): enables mixing of random sources from
            different tracks to assemble mix. Untested, it might contain bugs.
            Default to False.
        convert_to_mono (bool, optional): Convert the audio to mono. Default to True.
        train_minus_one (bool, optional): Return the targets as the minus on of the
            source instead as the separated stem. Default to False.
        source_augmentations (:obj:`list` of :obj:`callable`): list of augmentation
            function names, defaults to no-op augmentations (input = output)
        sample_rate (int, optional): Samplerate of files in dataset.
            It will resample the signals after loading them if they have a different
            sample rate (slow). Default to 44100.
        mix_only (bool, optional): Return only the mixture without the targets.
            Default to False.
        fake_musdb_format (bool, optional): Return the data in the format of the MUSDB18
            dataset to be used with the museval library. Default to False.
        size_limit (int/float, optional): Limit the number of tracks to load (if integer)
            or ratio of the dataset (if float between 0 and 1).
            Default to None (load all the tracks).
    """

    dataset_name = "EnsembleSet"

    def __init__(
        self,
        ensembleset_root_path,
        sources=('Violin', 'Viola', 'Cello', 'Bass', 'Flute', 'Clarinet', 'Oboe',
                 'Bassoon', 'Horn', 'Trumpet', 'Timpani'),
        targets=None,
        segment=None,
        samples_per_track=1,
        random_segments=False,
        fixed_segments=False,
        random_track_mix=False,
        convert_to_mono=True,
        train_minus_one=False,
        source_augmentations=lambda audio: audio,
        sample_rate=44100,
        mix_only=False,
        fake_musdb_format=False,
        size_limit=None,
    ):
        assert not (fixed_segments and random_track_mix)
        assert not (mix_only and fake_musdb_format)
        self.ensembleset_root_path = Path(ensembleset_root_path).expanduser()
        self.sample_rate = sample_rate
        self.segment = int(segment * sample_rate) if segment else None
        self.random_segments = random_segments
        self.fixed_segments = fixed_segments
        self.random_track_mix = random_track_mix
        self.convert_to_mono = convert_to_mono
        self.train_minus_one = train_minus_one
        self.source_augmentations = source_augmentations
        self.sources = sources
        self.sources = [source if source != 'Violin_1' else 'Violin'
                        for source in self.sources if source != 'Violin_2']
        self.targets = targets if targets is not None else sources
        self.targets = [target for target in self.targets if target in self.sources]
        self.targets = [target if target != 'Violin_1' else 'Violin'
                        for target in self.targets if target != 'Violin_2']
        self.samples_per_track = samples_per_track
        self.mix_only = mix_only
        self.fake_musdb_format = fake_musdb_format
        self.tracks = self.get_tracks()
        if not self.tracks:
            raise RuntimeError("No tracks found.")
        if fixed_segments:
            self.tracks_start = [None, ] * (len(self.tracks) * self.samples_per_track)
        if size_limit is not None:
            if size_limit < 1:
                size_limit = int(size_limit * len(self.tracks))
            self.tracks = self.tracks[:size_limit]

    def __getitem__(self, index):
        # load sources
        audio_sources = {}
        for source in self.sources:
            track_id = random.choice(range(len(self.tracks))) if self.random_track_mix \
                else index // self.samples_per_track
            start, end = self.get_track_segment(track_id)
            if source in self.tracks[track_id]:
                for file_idx in range(len(self.tracks[track_id][source])):
                    try:
                        audio, sr = sf.read(str(self.tracks[track_id][source][file_idx]), start=start, stop=end, always_2d=True)
                        if sr != self.sample_rate:
                            audio = librosa.resample(audio.T, orig_sr=sr, target_sr=self.sample_rate, res_type='polyphase').T
                    except RuntimeError:
                        print(f"Failed to open {Path(self.tracks[track_id][source][file_idx])} with start={start} and end={end}.")
                        print(f"Replacing the source by silence.")
                        audio = np.zeros((end-start, 2))
                    audio = audio.astype(np.float32).T
                    if self.convert_to_mono:
                        audio = np.mean(audio, axis=0, keepdims=True)
                    if source in audio_sources:
                        audio_sources[source] += self.source_augmentations(audio)
                    else:
                        audio_sources[source] = self.source_augmentations(audio)
            else:
                audio_sources[source] = np.zeros((1 if self.convert_to_mono else 2, end-start), dtype=np.float32)
        audio_mix = np.sum(list(audio_sources.values()), axis=0)

        if self.mix_only:
            return audio_mix
        else:
            if self.targets:
                audio_sources = np.stack([audio_sources[target] for target in self.targets], axis=0)
                if self.train_minus_one:
                    audio_sources = audio_mix - audio_sources
            if self.fake_musdb_format:
                sample_source = list(self.tracks[track_id].keys())[0]  # A source present in the track
                fake_musdb_track = SimpleNamespace()
                fake_musdb_track.name = Path(self.tracks[track_id][sample_source][0]).parts[-3]
                fake_musdb_track.folder = Path(self.tracks[track_id][sample_source][0]).parts[-3]
                fake_musdb_track.rate = self.sample_rate
                fake_musdb_track.subset = ""
                fake_musdb_track.audio = audio_mix.T
                fake_musdb_track.targets = {target_name: SimpleNamespace() for target_name in self.targets}
                for target_name, target_audio in zip(self.targets, audio_sources):
                    fake_musdb_track.targets[target_name].audio = target_audio.T
                return fake_musdb_track
            else:
                return audio_mix, audio_sources

    def __len__(self):
        return len(self.tracks) * self.samples_per_track

    def get_track_segment(self, track_idx):
        """Return the segment of the track to load"""
        track_info = sf.info(list(self.tracks[track_idx].values())[0][0])  # Get the path of the first source (all sources have the same length)
        track_len = track_info.frames
        if self.segment is not None and self.segment != 0:
            if self.random_segments:
                if self.fixed_segments:
                    if self.tracks_start[track_idx] is None:
                        self.tracks_start[track_idx] = int(random.uniform(0, track_len - self.segment))
                    start = self.tracks_start[track_idx]
                    end = start + self.segment
                else:
                    start = int(random.uniform(0, track_len - self.segment))
                    end = start + self.segment
            else:
                start = 0
                end = track_len
        else:
            start = 0
            end = track_len
        return start, end

    def get_tracks(self):
        """Return the path to the flac files of every source"""

        tracks = []
        for track_path in self.ensembleset_root_path.iterdir():
            track_path = track_path / 'Tree'
            if track_path.is_dir():
                sources_paths = {}
                for inst_path in track_path.iterdir():
                    if inst_path.suffix == ".flac":
                        for source in self.sources:
                            if source in inst_path.stem:
                                if source in sources_paths:
                                    sources_paths[source].append(inst_path)
                                else:
                                    sources_paths[source] = [inst_path]
                tracks.append(sources_paths)

        return tracks


class AaltoAnechoicOrchestralDataset(torch.utils.data.Dataset):
    """Aalto Anechoic Orchestral Dataset.

    Pytorch dataset for the Aalto Anechoic Orchestral recordings.
    Only for evaluation, not for training.

    https://research.cs.aalto.fi/acoustics/virtual-acoustics/research/acoustic-measurement-and-analysis/85-anechoic-recordings.html

    Rather than the original recordings, this class is designed for the denoised versions available in
    https://www.upf.edu/web/mtg/phenicx-anechoic
    """

    def __init__(self,
                 root_path,
                 sources=('Violin_1', 'Violin_2', 'Viola', 'Cello', 'Bass', 'Flute', 'Clarinet', 'Oboe', 'Bassoon',
                          'Horn', 'Trumpet', 'Trombone', 'Tuba', 'Harp', 'Timpani', 'untunedpercussion'),
                 targets=None,
                 join_violins=True,
                 sample_rate=44100,
                 ):

        self.sod2aalto = {'Bass': ['doublebass'],
                          'Bassoon': ['bassoon'],
                          'Cello': ['cello'],
                          'Clarinet': ['clarinet'],
                          'Flute': ['flute'],
                          'Harp': [],
                          'Horn': ['horn'],
                          'Oboe': ['oboe'],
                          'Timpani': [],
                          'Trombone': [],
                          'Trumpet': ['trumpet'],
                          'Tuba': [],
                          'Viola': ['viola'],
                          'Violin_1': ['violin'],
                          'Violin_2': [],
                          'Violin': ['violin'],
                          'untunedpercussion': [],
                         }

        self.sources = sources
        self.targets = targets if targets is not None else sources
        self.sources = [source if source != 'Violin_1' else 'Violin'
                        for source in self.sources if source != 'Violin_2']
        self.targets = [target if target != 'Violin_1' else 'Violin'
                        for target in self.targets if target != 'Violin_2']
        self.sources = {source: self.sod2aalto[source] for source in self.sources}
        self.targets = {target: self.sod2aalto[target] for target in self.targets}
        self.join_violins = join_violins
        self.sample_rate = sample_rate

        self.tracks = list(self.get_tracks(root_path + '/audio/'))
        if not self.tracks:
            raise RuntimeError("No tracks found.")

    def __getitem__(self, index):
        return self.tracks[index]

    def __len__(self):
        return len(self.tracks)

    def get_tracks(self, root_path):
        p = Path(root_path)

        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                fake_musdb_track = None
                for inst_path in track_path.iterdir():
                    if inst_path.suffix == ".wav":
                        signal, fs = sf.read(str(inst_path), always_2d=True)
                        if fs != self.sample_rate:
                            signal = librosa.resample(signal.T, orig_sr=fs, target_sr=self.sample_rate, res_type='polyphase').T
                        if fake_musdb_track is None:
                            fake_musdb_track = SimpleNamespace()
                            fake_musdb_track.folder = track_path.stem
                            fake_musdb_track.name = track_path.stem
                            fake_musdb_track.rate = fs
                            fake_musdb_track.audio = np.zeros_like(signal)
                            fake_musdb_track.instruments = []
                            fake_musdb_track.targets = {target_name: SimpleNamespace() for target_name in self.targets}
                            for target in fake_musdb_track.targets.values():
                                target.audio = np.zeros_like(signal)
                                target.instruments = []
                        assert fs == fake_musdb_track.rate
                        for source, instruments in self.sources.items():
                            if any([instrument in inst_path.stem for instrument in instruments]):
                                if len(signal) != len(fake_musdb_track.audio):
                                    # Only happens for Beethoven's double-bass
                                    signal = np.pad(signal, ((0, len(fake_musdb_track.audio) - len(signal)), (0, 0)))
                                fake_musdb_track.audio += signal
                                fake_musdb_track.instruments.append(inst_path.stem)
                                if source in self.targets:
                                    fake_musdb_track.targets[source].audio += signal
                                    fake_musdb_track.targets[source].instruments.append(inst_path.stem)
                if fake_musdb_track is not None:
                    peak = np.max(np.abs(fake_musdb_track.audio))
                    fake_musdb_track.audio /= (peak / 0.75)
                    for target in fake_musdb_track.targets.values():
                        target.audio /= (peak / 0.75)
                    yield fake_musdb_track


class URMPDataset(torch.utils.data.Dataset):
    """URMP Dataset

    Pytorch dataset for the audio recordings of the URMP dataset.
    Only for evaluation, not for training.

    https://labsites.rochester.edu/air/projects/URMP.html
    """

    def __init__(self,
                 root_path,
                 sources=('Violin_1', 'Violin_2', 'Viola', 'Cello', 'Bass', 'Flute', 'Clarinet', 'Oboe', 'Bassoon',
                          'Horn', 'Trumpet', 'Trombone', 'Tuba', 'Harp', 'Timpani', 'untunedpercussion'),
                 targets=None,
                 join_violins=True,
                 exclude_single_instrument_tracks=True,
                 exclude_saxophone_tracks=True,
                 sample_rate=44100,
                 ):

        self.sod2urmp = {'Bass': ['_db_'],
                         'Bassoon': ['_bn_'],
                         'Cello': ['_vc_'],
                         'Clarinet': ['_cl_'],
                         'Flute': ['_fl_'],
                         'Harp': [],
                         'Horn': ['_hn_'],
                         'Oboe': ['_ob_'],
                         'Timpani': [],
                         'Trombone': ['_tbn_'],
                         'Trumpet': ['_tpt_'],
                         'Tuba': ['_tba_'],
                         'Viola': ['_va_'],
                         'Violin_1': ['_vn_'],
                         'Violin_2': [],
                         'Violin': ['_vn_'],
                         'untunedpercussion': [],
                        }

        self.sources = sources
        self.targets = targets if targets is not None else sources
        if join_violins:
            self.sources = [source if source != 'Violin_1' else 'Violin'
                            for source in self.sources if source != 'Violin_2']
            self.targets = [target if target != 'Violin_1' else 'Violin'
                            for target in self.targets if target != 'Violin_2']
        self.sources = {source: self.sod2urmp[source] for source in self.sources}
        self.targets = {target: self.sod2urmp[target] for target in self.targets}
        self.join_violins = join_violins
        self.exclude_single_instrument_tracks = exclude_single_instrument_tracks
        self.exclude_saxophone_tracks = exclude_saxophone_tracks
        self.sample_rate = sample_rate

        self.tracks = list(self.get_tracks(root_path))
        if not self.tracks:
            raise RuntimeError("No tracks found.")

    def __getitem__(self, index):
        return self.tracks[index]

    def __len__(self):
        return len(self.tracks)

    def get_tracks(self, root_path):
        p = Path(root_path)

        for track_path in tqdm.tqdm(p.iterdir()):
            if track_path.is_dir():
                if self.exclude_saxophone_tracks and '_sax' in track_path.stem:
                    continue
                fake_musdb_track = None
                for inst_path in track_path.iterdir():
                    if inst_path.stem[:6] == "AuSep_" and inst_path.suffix == ".wav":
                        signal, fs = sf.read(str(inst_path), always_2d=True)
                        if fs != self.sample_rate:
                            signal = librosa.resample(signal.T, orig_sr=fs, target_sr=self.sample_rate, res_type='polyphase').T
                        if fake_musdb_track is None:
                            fake_musdb_track = SimpleNamespace()
                            fake_musdb_track.folder = track_path.stem
                            fake_musdb_track.name = track_path.stem
                            fake_musdb_track.rate = fs
                            fake_musdb_track.audio = np.zeros_like(signal)
                            fake_musdb_track.instruments = []
                            fake_musdb_track.targets = {target_name: SimpleNamespace() for target_name in self.targets}
                            for target in fake_musdb_track.targets.values():
                                target.audio = np.zeros_like(signal)
                                target.instruments = []
                        assert fs == fake_musdb_track.rate
                        for source, instruments in self.sources.items():
                            if any([instrument in inst_path.stem for instrument in instruments]):
                                if len(signal) != len(fake_musdb_track.audio):
                                    # Only happens for Beethoven's double-bass
                                    # signal = np.pad(signal, ((0, len(fake_musdb_track.audio) - len(signal)), (0, 0)))
                                    pass
                                fake_musdb_track.audio += signal
                                fake_musdb_track.instruments.append(inst_path.stem)
                                if source in self.targets:
                                    fake_musdb_track.targets[source].audio += signal
                                    fake_musdb_track.targets[source].instruments.append(inst_path.stem)
                if fake_musdb_track is None or (self.exclude_single_instrument_tracks and len(fake_musdb_track.instruments) == 1):
                    continue
                if fake_musdb_track is not None:
                    peak = np.max(np.abs(fake_musdb_track.audio))
                    fake_musdb_track.audio /= (peak / 0.75)
                    for target in fake_musdb_track.targets.values():
                        target.audio /= (peak / 0.75)
                    yield fake_musdb_track


def load_synthsod_datasets(parser, args):
    """Loads the SynthSOD dataset from commandline arguments.

    Returns:
        train_dataset, validation_dataset
    """

    args = parser.parse_args()

    source_augmentations = Compose(
        [globals()["_augment_" + aug] for aug in args.source_augmentations]
    )

    train_dataset = SynthSODDataset(
        metadata_file_path=args.synthsod_dataset_path + '/SynthSOD_metadata_train.json',
        synthsod_data_path=args.synthsod_dataset_path + '/SynthSOD_data/',
        sources=args.sources,
        targets=args.targets,
        join_violins=args.join_violins,
        convert_to_mono=(args.nb_channels==1),
        source_augmentations=source_augmentations,
        random_track_mix=args.random_track_mix,
        segment=args.seq_dur,
        random_segments=True,
        sample_rate=args.sample_rate,
        samples_per_track=args.samples_per_track,
        size_limit=args.train_size_limit,
        train_minus_one=args.train_minus_one,
    )

    valid_dataset = SynthSODDataset(
        metadata_file_path=args.synthsod_dataset_path + '/SynthSOD_metadata_evaluation.json',
        synthsod_data_path=args.synthsod_dataset_path + '/SynthSOD_data/',
        sources=args.sources,
        targets=args.targets,
        join_violins=args.join_violins,
        convert_to_mono=(args.nb_channels==1),
        segment=args.seq_dur,
        random_segments=True,
        fixed_segments=True,
        sample_rate=args.sample_rate,
        samples_per_track=args.samples_per_track,
        train_minus_one=args.train_minus_one,
    )

    return train_dataset, valid_dataset


def load_ensembleset_datasets(parser, args):
    """Loads the EnsembleSet dataset from commandline arguments.
    Note that EnsembleSet does not contain train/evaluation/test splits, so
    the same dataset is returned for both train and validation.

    Returns:
        train_dataset, validation_dataset
    """

    args = parser.parse_args()

    source_augmentations = Compose(
        [globals()["_augment_" + aug] for aug in args.source_augmentations]
    )

    train_dataset = EnsembleSetDataset(
        ensembleset_root_path=args.ensembleset_dataset_path + '/BBCSO_Ensembles/',
        sources=args.sources,
        targets=args.targets,
        convert_to_mono=(args.nb_channels==1),
        source_augmentations=source_augmentations,
        random_track_mix=args.random_track_mix,
        segment=args.seq_dur,
        random_segments=True,
        sample_rate=args.sample_rate,
        samples_per_track=args.samples_per_track,
        size_limit=args.train_size_limit,
        train_minus_one=args.train_minus_one,
    )

    valid_dataset = EnsembleSetDataset(
        ensembleset_root_path=args.ensembleset_dataset_path + '/BBCSO_Ensembles/',
        sources=args.sources,
        targets=args.targets,
        convert_to_mono=(args.nb_channels==1),
        segment=args.seq_dur,
        random_segments=True,
        fixed_segments=True,
        sample_rate=args.sample_rate,
        samples_per_track=args.samples_per_track,
        train_minus_one=args.train_minus_one,
    )

    return train_dataset, valid_dataset


class Compose(object):
    """Composes several augmentation transforms.
    Literally taken from the original one on the asteroid MUSDB18 example.

    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for transform in self.transforms:
            audio = transform(audio)
        return audio


def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain to each source between `low` and `high`
     Literally taken from the original one on the asteroid MUSDB18 example."""
    gain = low + np.random.rand(1).astype(np.float32) * (high - low)
    return audio * gain


def _augment_channelswap(audio):
    """Randomly swap channels of stereo sources
     Literally taken from the original one on the asteroid MUSDB18 example."""
    if audio.shape[0] == 2 and np.random.rand(1) < 0.5:
        return np.flip(audio, [0])

    return audio

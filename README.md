# SynthSOD-Baseline

Repository with the code for training and evaluating the baseline model of the SynthSOD dataset [[1]](#references).

The code is based on the MUSDB18 example of Asteroid for X-UMX, but with some modifications to
make it work for orchestra music. It supports training on SynthSOD and EnsembleSet and evaluation
in those and also in the Aalto anechoic orchestra recorgins and on the URMP ensembles dataset.

The SynthSOD dataset can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.13759492) 
and the code to generate it from the original [SOD MIDI files](https://qsdfo.github.io/LOP/database.html) 
is also available in [GitHub](https://github.com/repertorium/HQ-SOD-generator).

- [Dependencies](#dependencies)
- [Data](#data)
- [Training](#training)
- [Evaluation](#evaluation)
- [Pretrained models](#pretrained-models)
- [Known issues](#known-issues)
- [License](#license)
- [References](#references)

## Dependencies

You will need to install the following dependencies to train and evaluate the models:

* [**asteroid**](https://github.com/asteroid-team/asteroid/): The model X-UMX is not working on the 
current version available on pip (0.7.0) but it is already fixed in the master branch of their GitHub
repository. You can install it with `pip install https://github.com/asteroid-team/asteroid/zipball/master`.
* [**museval**](https://github.com/sigsep/sigsep-mus-eval): The original version of the library does not support
tracks with completely silent instruments, which is a quite common situation in SynthSOD, EnsembleSet, or any
other classical-music dataset where not all the tracks contain all the instruments. We have submited a Pull Request
fixing this, but in case it has not been approved yet, you can download our fork with 
`pip install https://github.com/DavidDiazGuerra/sigsep-mus-eval/zipball/allow-silents`

## Data

You should download the followoing datasets in the `data` folder to train or evaluate the models on them:

* [**SynthSOD**](https://doi.org/10.5281/zenodo.13759492): Our model for orchestra music separation, containing almost 50 hours of ensemble and orchestra
music synthesized with the Spitfire's BBC Orchestra Professional VST plugin.
* [**EnsembleSet**](https://zenodo.org/records/7327175): A dataset with about 6 hours of ensemble music synthesized 
with the Spitfire's BBC Orchestra Professional VST plugin.
* [**Aalto anechoic orchestra recordings**](https://research.cs.aalto.fi/acoustics/virtual-acoustics/research/acoustic-measurement-and-analysis/85-anechoic-recordings.html): 
A dataset with about 10 minutes of orchestra music with every instrument recorded isolately in an anechoic chamber.
Instead of the original recordings, we use the denoised version from the [PHENICX project](https://www.upf.edu/web/mtg/phenicx-anechoic).
* [**URMP**](https://labsites.rochester.edu/air/projects/URMP.html): A dataset with about 1 hour of ensemble 
music with the instruments recorded isolately.

You will need to download the datasets and place them in the `data` folder.

## Training

Training a single X-UMX model for 15 output instruments needs a huge amount of GPU memory, so we trained 4 independent
models, each one of them for the instruments of one family of instruments: strings (violin, viola, cello, and bass), 
woodwinds (flute, clarinet, oboe, and bassoon), brass (horn, trumpet, trombone, and tuba), and percussion (harp, 
timpani, and untuned percussion). We provide 4 bash script to train the models:

```bash
run_training_string.sh
run_training_woodwind.sh
run_training_brass.sh
run_training_percussion.sh
```

These scripts also allow to train the models on the EnsembleSet dataset by calling them with the following arguments:

```bash
run_training_strings.sh --train_on EnsembleSet --tag ensembleset_strings
run_training_woodwind.sh --train_on EnsembleSet --tag ensembleset_woodwind
run_training_brass.sh --train_on EnsembleSet --tag ensembleset_brass
run_training_percussion.sh --train_on EnsembleSet --tag ensembleset_percussion
```

## Evaluation

You can evaluate the models on the datasets by running the following script:

```bash
run_evaluation.sh
```

If you want to evaluate the models trained on EnsembleSet, you can call the script with the following arguments:

```bash
run_evaluation.sh --tag ensembleset
```

## Pretrained models

You can find the pretrained models that obtained the previous results in the `Releases` section of this repository.

## Known issues

During the training, you might get some error messages similar to `Failed to open 
data/SynthSOD/SynthSOD_data/symphony_7_2_orch/Tree/Violin_2.flac with start=13851056 and end=14115656`, which
are probably due to the exception `LibsndfileError: Internal psf_fseek() failed` raising when trying to read some
audio files with the `soundfile` library. These errors depend on the specific segment that is being read (reading
the whole audio file does not raise any error) and on the machine where the training is running (the same audio file
might not raise any problems in other machines) so they are quite difficult to debug. 

Any further insights about what can be causing these errors or how to fix them are welcome, but as long as they do not
happen too often, they should not have a big impact on the training process. Actually, the pretrained models provided 
in the repository were trained with some of these errors happening.

## License

The software is subject to AGPL-3.0 license and comes with no warranty. If you find it useful for your research work, please, acknowledge it to [[1]](#references).

## References

[1] Garcia-Martinez J., Diaz-Guerra, D., Politis, A., Virtanen, T., Carabias-Orti, J.J., & Vera-Candeas, P. SynthSOD: Developing an Heterogeneous Dataset for Orchestra Music Source Separation [[arXiv preprint](https://doi.org/10.48550/arXiv.2409.10995)]

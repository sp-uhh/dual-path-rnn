# Dual-Path RNN for Single-Channel Speech Separation (in Keras-Tensorflow2)

Keras-Tensorflow2 implementation of Dual-Path RNN as in [1] for Speech Separation trained on WSJ0-MIX2 subset of the WHAM! data set.

This implementation achieves 14.5 dB SI-SNR improvement on the WSJ0-MIX2 subset of the WHAM! [2] test set.

Please give credit to this Github repository when using this implementation for publications.

References:

[1] Yi Luo, Zhuo Chen, Takuya Yoshioka, "Dual-path RNN: efficient long sequence modeling for time-domain single-channel speech separation", arXiv preprint arXiv:1910.06379 2019. Available under https://arxiv.org/abs/1910.06379

[2] Wichern, G., Antognini, J., Flynn, M., Zhu, L.R., McQuinn, E., Crow, D., Manilow, E., Roux, J.L. (2019) WHAM!: Extending Speech Separation to Noisy Environments. Proc. Interspeech 2019, 1368-1372, DOI: 10.21437/Interspeech.2019-2821. Available under https://www.isca-speech.org/archive/Interspeech_2019/abstracts/2821.html

## Requirements

Resolve Python package dependencies with:

- pip install numpy librosa tensorflow-gpu

GPU requirements in default configuration:

- Tested with 11 GB RAM GPU (NVidia GeForce RTX 2080 Ti)

Data requirements:

- WHAM data set. Needs commercial WSJ0 data set (https://catalog.ldc.upenn.edu/LDC93S6B) as basis. Scripts and noise data for WHAM data set publicly availabe at http://wham.whisper.ai/


## How to start experiment:
1.  Clone repo and go to src/
2.  In run_experiment.py modify path to WHAM data and possibly change further parameters.
3.  Run `CUDA_VISIBLE_DEVICES=0 python run_experiment.py` where 0 is the ID of your GPU Device.
4.  Run `tensorboard --logdir exp/EXPERIMENT_TAG/tensorboard_logs` and open the URL printed by tensorbard. You can watch experiment status and for each epoch some audio files will be generated in tab "audio".
4.  Find training and separation output in exp/ directory. A copy of the configuration file is also stored in this directory.

Roughly 1 hour per epoch and stopping criterion was reached after 102 epochs (~5 days total training time) tested on a single NVidia GeForce RTX 2080 Ti. You can find the model from this training under exp/pretrained .

## Shortcomings regarding [1]
* Due to limited GPU resources, we used 3 DPRNN blocks instead of 6 proposed in [1]. If batch size is changed to 1 instead of 2, 6 DPRRN blocks can be fit in a 11 GB RAM GPU but experiments have shown that this minimal batch size deteriorates the overall performance.
* Activating the scaling factors in normalization layers as proposed in [1] adds too many weights to the network. For this reason, we turned scaling off. It is unclear if the Tensorflow implementation of the global layer normalization is different from the implementation used in [1].

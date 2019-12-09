import sys
import os
import numpy as np
from shutil import copyfile
from stat import S_IREAD, S_IRGRP, S_IROTH

from network import TasnetWithDprnn
from train import train_network
from separate import Separator
from evaluate import Evaluator

WHAM_ROOT_DIR = '/data1/ditter/speechSeparation/preprocessedData/wham/'

FILE_LIST_PATH_TRAIN = os.path.join(WHAM_ROOT_DIR, 'create-speaker-mixtures', 'mix_2_spk_min_' + 'tr' + '_mix')
FILE_LIST_PATH_VALID = os.path.join(WHAM_ROOT_DIR, 'create-speaker-mixtures', 'mix_2_spk_min_' + 'cv' + '_mix')
FILE_LIST_PATH_TEST = os.path.join(WHAM_ROOT_DIR, 'create-speaker-mixtures', 'mix_2_spk_min_' + 'tt' + '_mix')

WAV_DIR_TRAIN = os.path.join(WHAM_ROOT_DIR, 'wav8k', 'min', 'tr')
WAV_DIR_VALID = os.path.join(WHAM_ROOT_DIR, 'wav8k', 'min', 'cv')
WAV_DIR_TEST = os.path.join(WHAM_ROOT_DIR, 'wav8k', 'min', 'tt')
SAMPLERATE_HZ = 8000

EXPERIMENT_ROOT_DIR = '../exp/'
EXPERIMENT_TAG = 'default'

RESUME_TRAINING = False
RESUME_FROM_EPOCH = 10
RESUME_FROM_MODEL_DIR = 'default'

# TRAINING PARAMETERS
BATCH_SIZE = 2
NUM_BATCHES_TRAIN = 20000 // BATCH_SIZE
NUM_BATCHES_VALID = 400 // BATCH_SIZE
NUM_EPOCHS = 200
NUM_EPOCHS_FOR_EARLY_STOPPING = 10
OPTIMIZER_CLIP_L2_NORM_VALUE = 5
TRAIN_UTTERANCE_LENGTH_IN_SECONDS = 4
SEPARATE_MAX_UTTERANCE_LENGTH_IN_SECONDS = 14 # WHAM! test set longest WAV file length is 13.4 seconds

# NETWORK PARAMETERS
NETWORK_NUM_FILTERS_IN_ENCODER = 64
NETWORK_ENCODER_FILTER_LENGTH = 2
NETWORK_NUM_UNITS_PER_LSTM = 200
NETWORK_NUM_DPRNN_BLOCKS = 3
NETWORK_CHUNK_SIZE = 256

def run_experiment(stage=0):
    np.random.seed(0)
    experiment_dir = os.path.join(EXPERIMENT_ROOT_DIR, EXPERIMENT_TAG)
    wav_output_dir = os.path.join(experiment_dir, 'separate')
    validation_loss_file = os.path.join(experiment_dir, 'validation_loss.npy.txt')
    train_num_full_chunks = SAMPLERATE_HZ*TRAIN_UTTERANCE_LENGTH_IN_SECONDS//NETWORK_CHUNK_SIZE
    separate_max_num_full_chunks = SAMPLERATE_HZ*SEPARATE_MAX_UTTERANCE_LENGTH_IN_SECONDS//NETWORK_CHUNK_SIZE

    if stage <= 0: # Start with training
        if os.path.exists(experiment_dir):
            sys.exit('Experiment tag already in use. Change tag and run again')
        os.mkdir(experiment_dir)
        config_backup_file = os.path.join(experiment_dir, 'config.py')
        copyfile(os.path.realpath(__file__), config_backup_file)
        os.chmod(config_backup_file, S_IREAD|S_IRGRP|S_IROTH)

        if RESUME_TRAINING:
            model_weights_file = os.path.join(RESUME_FROM_MODEL_DIR, \
                                              'state_epoch_' + str(RESUME_FROM_EPOCH) + '.h5')
        else:
            model_weights_file = None

        # Generate network
        tasnet = TasnetWithDprnn(batch_size=BATCH_SIZE,
                                 model_weights_file=model_weights_file,
                                 num_filters_in_encoder=NETWORK_NUM_FILTERS_IN_ENCODER,
                                 encoder_filter_length=NETWORK_ENCODER_FILTER_LENGTH,
                                 chunk_size=NETWORK_CHUNK_SIZE,
                                 num_full_chunks=train_num_full_chunks,
                                 units_per_lstm=NETWORK_NUM_UNITS_PER_LSTM,
                                 num_dprnn_blocks=NETWORK_NUM_DPRNN_BLOCKS,
                                 samplerate_hz=SAMPLERATE_HZ)

        # Train network
        tensorboard_dir = os.path.join(experiment_dir, 'tensorboard_logs')
        print('Run follwing command to run Tensorboard: \n', 'tensorboard --bind_all --logdir ' + tensorboard_dir)
        validation_loss = train_network(experiment_dir=experiment_dir,
                                        tensorboard_dir=tensorboard_dir,
                                        batch_size=BATCH_SIZE,
                                        num_batches_train=NUM_BATCHES_TRAIN,
                                        num_batches_valid=NUM_BATCHES_VALID,
                                        num_epochs=NUM_EPOCHS,
                                        num_epochs_for_early_stopping=NUM_EPOCHS_FOR_EARLY_STOPPING,
                                        optimizer_clip_l2_norm_value=OPTIMIZER_CLIP_L2_NORM_VALUE,
                                        samplerate_hz=SAMPLERATE_HZ,
                                        utterance_length_in_seconds=TRAIN_UTTERANCE_LENGTH_IN_SECONDS,
                                        wav_data_dir_train=WAV_DIR_TRAIN,
                                        wav_data_dir_valid=WAV_DIR_VALID,
                                        file_list_path_train=FILE_LIST_PATH_TRAIN,
                                        file_list_path_valid=FILE_LIST_PATH_VALID,
                                        tasnet=tasnet)
        np.savetxt(validation_loss_file, validation_loss, fmt="%.2f")

    if stage <= 1: # Start with separation
        if os.path.exists(wav_output_dir):
            sys.exit('Separation folder already exists')
        os.mkdir(wav_output_dir)
        validation_loss_per_epoch = np.loadtxt(validation_loss_file)
        epoch_with_best_validation_result = np.argmin(validation_loss_per_epoch) + 1
        model_weights_file = os.path.join(experiment_dir, 'state_epoch_' \
                                          + str(epoch_with_best_validation_result) + '.h5')

        # Generate trained network
        tasnet = TasnetWithDprnn(batch_size=1,
                                 model_weights_file=model_weights_file,
                                 num_filters_in_encoder=NETWORK_NUM_FILTERS_IN_ENCODER,
                                 encoder_filter_length=NETWORK_ENCODER_FILTER_LENGTH,
                                 chunk_size=NETWORK_CHUNK_SIZE,
                                 num_full_chunks=separate_max_num_full_chunks,
                                 units_per_lstm=NETWORK_NUM_UNITS_PER_LSTM,
                                 num_dprnn_blocks=NETWORK_NUM_DPRNN_BLOCKS,
                                 samplerate_hz=SAMPLERATE_HZ)

        # Use network to separate list of wav files
        separator = Separator(tasnet=tasnet,
                              input_dir=os.path.join(WAV_DIR_TEST, 'mix_clean'),
                              output_dir=wav_output_dir,
                              max_num_chunks=separate_max_num_full_chunks)
        separator.process_file_list(FILE_LIST_PATH_TEST)

    if stage <= 2:  # Start with evaluation

        # Evaluate list of separated wav files
        evaluator = Evaluator(estimate_wav_dir=wav_output_dir,
                              groundtruth_wav_dir=WAV_DIR_TEST,
                              sample_list_path=FILE_LIST_PATH_TEST)
        print('SI-SNR Performance on Test Set:', evaluator.mean_sisnr)
        np.savetxt(os.path.join(experiment_dir, 'results.npy.txt'), evaluator.results, fmt="%.2f")
        np.savetxt(os.path.join(experiment_dir, 'mean_result.npy.txt'), np.array([evaluator.mean_sisnr]), fmt="%.2f")


if len(sys.argv) > 1:
    STAGE = int(sys.argv[1])
else:
    STAGE = 0

run_experiment(stage=STAGE)

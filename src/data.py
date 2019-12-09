import os
import numpy as np
from librosa import load
from librosa.util import fix_length

from utils import text_file_2_list

# This class enables data handling for the training and evaluation processes.
class TasnetData():
    def __init__(self, data_root_dir, file_list_path, samplerate_hz,
                 utterance_length_in_seconds, num_speakers):
        self.data_root_dir = data_root_dir
        self.file_list_path = file_list_path
        self.samplerate_hz = samplerate_hz
        self.num_speakers = num_speakers
        self.utterance_length_in_seconds = utterance_length_in_seconds
        self.samples_per_utterance = int(samplerate_hz*utterance_length_in_seconds)

        self.file_list = text_file_2_list(file_list_path)

    def collect_data_for_one_sample(self, sample_name):
        # Get root dirs
        unprocessed_dir = os.path.join(self.data_root_dir, 'mix_clean')
        groundtruth_spk0_dir = os.path.join(self.data_root_dir, 's1')
        groundtruth_spk1_dir = os.path.join(self.data_root_dir, 's2')

        # Get full file paths
        unprocessed_file = os.path.join(unprocessed_dir, sample_name + '.wav')
        groundtruth_spk0_file = os.path.join(groundtruth_spk0_dir, sample_name + '.wav')
        groundtruth_spk1_file = os.path.join(groundtruth_spk1_dir, sample_name + '.wav')

        # Collect signals
        unprocessed, _ = load(unprocessed_file, sr=self.samplerate_hz)
        groundtruth_spk0, _ = load(groundtruth_spk0_file, sr=self.samplerate_hz)
        groundtruth_spk1, _ = load(groundtruth_spk1_file, sr=self.samplerate_hz)

        # Determine start point of segment
        if unprocessed.size > self.samples_per_utterance:
            max_shift = unprocessed.size - self.samples_per_utterance
            start_point = np.random.randint(max_shift)
        else:
            start_point = 0

        # Cut segment out segment
        unprocessed = fix_length(unprocessed[start_point:], self.samples_per_utterance)
        groundtruth_spk0 = fix_length(groundtruth_spk0[start_point:], self.samples_per_utterance)
        groundtruth_spk1 = fix_length(groundtruth_spk1[start_point:], self.samples_per_utterance)

        groundtruth = np.stack([groundtruth_spk0, groundtruth_spk1], axis=0)

        return unprocessed, groundtruth

    def collect_data(self, indices):
        num_files = len(indices)
        mixture = np.zeros((num_files, self.samples_per_utterance))
        groundtruth = np.zeros((num_files, self.num_speakers, self.samples_per_utterance))
        for i in range(num_files):
            sample_name = self.file_list[indices[i]]
            mixture[i, :], groundtruth[i, :, :] = self.collect_data_for_one_sample(sample_name)

        return mixture, groundtruth

    def batch_generator(self, batch_size, num_batches):
        shuffled_indices = np.arange(batch_size*num_batches)
        np.random.shuffle(shuffled_indices)
        shuffled_indices = np.reshape(shuffled_indices, (num_batches, batch_size))

        while 1:
            for i in range(num_batches):
                mixture, groundtruth = self.collect_data(shuffled_indices[i, :])
                yield(mixture, groundtruth)

import os
import librosa
import numpy as np

from loss import get_permutation_invariant_sisnr
from utils import text_file_2_list

# This class allows to evaluate the SI-SNR performance measures for one or a list of WAV files
class Evaluator():
    def __init__(self, estimate_wav_dir, groundtruth_wav_dir, sample_list_path):
        self.estimate_wav_dir = estimate_wav_dir
        self.groundtruth_wav_dir = groundtruth_wav_dir
        self.sample_list_path = sample_list_path

        self.evaluate_list(sample_list_path)
        self.mean_sisnr = np.mean(self.results)

    def evaluate_single_sample(self, sample_name):
        spk0_estimate_file = os.path.join(self.estimate_wav_dir, 's1', sample_name + '.wav')
        spk1_estimate_file = os.path.join(self.estimate_wav_dir, 's2', sample_name + '.wav')
        spk0_grpundtruth_file = os.path.join(self.groundtruth_wav_dir, 's1', sample_name + '.wav')
        spk1_groundtruth_file = os.path.join(self.groundtruth_wav_dir, 's2', sample_name + '.wav')

        spk0_estimate, _ = librosa.load(spk0_estimate_file, sr=None)
        spk1_estimate, _ = librosa.load(spk1_estimate_file, sr=None)
        spk0_groundtruth, _ = librosa.load(spk0_grpundtruth_file, sr=None)
        spk1_groundtruth, _ = librosa.load(spk1_groundtruth_file, sr=None)

        return get_permutation_invariant_sisnr(spk0_estimate, spk1_estimate, spk0_groundtruth, spk1_groundtruth)

    def evaluate_list(self, sample_list_path):
        sample_list = text_file_2_list(sample_list_path)
        num_files = len(sample_list)
        results = np.zeros((num_files, 2))
        for i in range(num_files):
            results[i, 0], results[i, 1] = self.evaluate_single_sample(sample_list[i])
            print(i+1, results[i, 0], results[i, 1])
        self.results = results

    def save_results(self, file_path):
        np.savetxt(file_path, self.results, fmt="%2.1f")

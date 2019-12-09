import os
import numpy as np
from librosa import load
from librosa.output import write_wav
from librosa.util import fix_length

from utils import text_file_2_list

# This class allows to separate speaker signals for one or a list of WAV files
class Separator():
    def __init__(self, tasnet, input_dir, output_dir, max_num_chunks):
        self.input_dir = input_dir
        self.tasnet = tasnet
        self.samplerate_hz = self.tasnet.samplerate_hz
        self.output_dir = output_dir
        self.max_length = max_num_chunks*self.tasnet.chunk_size

        dir_s1 = os.path.join(output_dir, 's1')
        dir_s2 = os.path.join(output_dir, 's2')
        if not os.path.exists(dir_s1):
            os.mkdir(dir_s1)
        if not os.path.exists(dir_s2):
            os.mkdir(dir_s2)

    def separate_single_mixture(self, mixture):
        original_length = mixture.size
        mixture_padded = fix_length(mixture, self.max_length)
        speaker_signals_padded = self.tasnet.model.predict(np.expand_dims(mixture_padded, axis=0))
        speaker_signals = fix_length(speaker_signals_padded[0, :, :], original_length, axis=1)
        return speaker_signals

    def process_single_file(self, file_name):
        mixture, _ = load(os.path.join(self.input_dir, file_name + '.wav'), sr=self.samplerate_hz)
        speaker_signals = self.separate_single_mixture(mixture)

        write_wav(os.path.join(self.output_dir, 's1', file_name + '.wav'), \
                  speaker_signals[0, :], self.samplerate_hz, norm=True)
        write_wav(os.path.join(self.output_dir, 's2', file_name + '.wav'), \
                  speaker_signals[1, :], self.samplerate_hz, norm=True)

    def process_file_list(self, wav_file_list):
        wav_list = text_file_2_list(wav_file_list)
        num_files = len(wav_list)
        for i, wav_file in enumerate(wav_list):
            self.process_single_file(wav_file)
            if i % 50 == 0:
                print(i, '/', num_files)

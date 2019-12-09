import os
import numpy as np
import tensorflow as tf
from tensorflow import keras

from data import TasnetData
from loss import permutation_invariant_loss

# This file holds the implementation of the training and logging procedures.
# Note that the network and loss are implemented in separate files.

NUM_SPEAKERS = 2 # Cannot be changed
NUM_LOG_AUDIO_FILES_PER_EPOCH = 10

INITIAL_LEARNING_RATE = 0.001
LEARNING_RATE_DECAY = 0.98
LEARING_RATE_DECAY_EVERY_N_EPOCHS = 2

def scheduler(epoch, learning_rate):
    if epoch > 0:
        if epoch % LEARING_RATE_DECAY_EVERY_N_EPOCHS == 0:
            learning_rate = learning_rate*LEARNING_RATE_DECAY
            print('Change learning rate to', "{0:.6f}".format(learning_rate))
    return learning_rate

class WriteValidationAudio(keras.callbacks.Callback):
    def __init__(self, log_dir, tasnet_data, n_utterances, model, samplerate_hz, batch_size, utterance_length_in_seconds):
        super(WriteValidationAudio, self).__init__()
        self.log_dir = log_dir
        self.n_utterances = n_utterances
        self.model = model
        self.samplerate_hz = samplerate_hz
        self.utterance_length_in_samples = utterance_length_in_seconds*samplerate_hz
        self.batch_size = batch_size
        self._total_batches_seen = 0
        self.input_mixtures, self.groundtruth = tasnet_data.collect_data(np.arange(n_utterances))

    def on_epoch_end(self, epoch, logs=None):
        output = self.model.predict(self.input_mixtures, batch_size=self.batch_size)
        output = output / np.amax(output, axis=2, keepdims=True) # Normalize

        valid_summary_writer = tf.summary.create_file_writer(self.log_dir)
        with valid_summary_writer.as_default():
            tf.summary.audio(f'{epoch}_audio', np.reshape(output, (self.n_utterances*NUM_SPEAKERS, self.utterance_length_in_samples, 1)).astype(np.float32),
                             self.samplerate_hz, encoding='wav', step=self._total_batches_seen, max_outputs=NUM_LOG_AUDIO_FILES_PER_EPOCH)

    def on_batch_end(self, batch, logs=None):
        self._total_batches_seen += 1

def train_network(experiment_dir,
                  tensorboard_dir,
                  batch_size,
                  num_batches_train,
                  num_batches_valid,
                  num_epochs,
                  num_epochs_for_early_stopping,
                  optimizer_clip_l2_norm_value,
                  samplerate_hz,
                  utterance_length_in_seconds,
                  wav_data_dir_train,
                  wav_data_dir_valid,
                  file_list_path_train,
                  file_list_path_valid,
                  tasnet):

    tasnet_data_train = TasnetData(data_root_dir=wav_data_dir_train,
                                   file_list_path=file_list_path_train,
                                   samplerate_hz=samplerate_hz,
                                   utterance_length_in_seconds=utterance_length_in_seconds,
                                   num_speakers=NUM_SPEAKERS)
    tasnet_data_valid = TasnetData(data_root_dir=wav_data_dir_valid,
                                   file_list_path=file_list_path_valid,
                                   samplerate_hz=samplerate_hz,
                                   utterance_length_in_seconds=utterance_length_in_seconds,
                                   num_speakers=NUM_SPEAKERS)

    train_generator = tasnet_data_train.batch_generator(batch_size=batch_size, num_batches=num_batches_train)
    validation_generator = tasnet_data_valid.batch_generator(batch_size=batch_size, num_batches=num_batches_valid)

    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=os.path.join(tensorboard_dir),
                                                       update_freq='batch',
                                                       write_graph=True)

    model_save_callbback = keras.callbacks.ModelCheckpoint(filepath=os.path.join(experiment_dir, 'state_epoch_{epoch}.h5'),
                                                           save_freq='epoch',
                                                           save_weights_only=True,
                                                           load_weights_on_restart=True)

    store_audio_callback = WriteValidationAudio(log_dir=os.path.join(tensorboard_dir),
                                                tasnet_data=tasnet_data_valid,
                                                n_utterances=num_batches_valid*batch_size,
                                                model=tasnet.model,
                                                batch_size=batch_size,
                                                samplerate_hz=samplerate_hz,
                                                utterance_length_in_seconds=utterance_length_in_seconds)

    learning_rate_callback = keras.callbacks.LearningRateScheduler(scheduler)
    early_stopping_callback = keras.callbacks.EarlyStopping(patience=num_epochs_for_early_stopping)

    adam = keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE, clipnorm=optimizer_clip_l2_norm_value)
    tasnet.model.compile(loss=permutation_invariant_loss, optimizer=adam)
    history = tasnet.model.fit_generator(train_generator,
                                         epochs=num_epochs,
                                         steps_per_epoch=num_batches_train,
                                         validation_data=validation_generator,
                                         validation_steps=num_batches_valid,
                                         validation_freq=1,
                                         callbacks=[tensorboard_callback,
                                                    model_save_callbback,
                                                    store_audio_callback,
                                                    learning_rate_callback,
                                                    early_stopping_callback])
    return history.history['val_loss']

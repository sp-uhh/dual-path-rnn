import tensorflow as tf
from tensorflow import keras

# This file hold implementation of the overall network structure as a keras model
# including the DPRNN blocks as part of this structure.

class DprnnBlock(keras.layers.Layer):
    def __init__(self, num_outputs, is_last_dprnn, tasnet_with_dprnn, **kwargs):
        super(DprnnBlock, self).__init__(**kwargs)
        self.num_outputs = num_outputs
        self.is_last_dprnn = is_last_dprnn

        # Copy relevant fields from Tasnet object
        self.batch_size = tasnet_with_dprnn.batch_size
        self.num_overlapping_chunks = tasnet_with_dprnn.num_overlapping_chunks
        self.chunk_size = tasnet_with_dprnn.chunk_size
        self.num_filters_in_encoder = tasnet_with_dprnn.num_filters_in_encoder
        self.units_per_lstm = tasnet_with_dprnn.units_per_lstm

        if is_last_dprnn:
            self.fc_units = self.num_filters_in_encoder*tasnet_with_dprnn.num_speakers
        else:
            self.fc_units = self.num_filters_in_encoder

        self.intra_rnn = keras.layers.Bidirectional(keras.layers.LSTM(units=self.units_per_lstm//2, return_sequences=True))
        self.intra_fc = keras.layers.Dense(units=self.num_filters_in_encoder)
        self.intra_ln = keras.layers.LayerNormalization(center=False, scale=False)

        self.inter_rnn = keras.layers.Bidirectional(keras.layers.LSTM(units=self.units_per_lstm//2, return_sequences=True))
        self.inter_fc = keras.layers.Dense(units=self.fc_units)
        self.inter_ln = keras.layers.LayerNormalization(center=False, scale=False)

    def call(self, T):
        # Intra-Chunk Processing
        T_shaped = tf.reshape(T, (self.batch_size*self.num_overlapping_chunks, self.chunk_size, self.num_filters_in_encoder))
        U = self.intra_rnn(T_shaped)
        U = tf.reshape(U, (self.batch_size*self.num_overlapping_chunks*self.chunk_size, self.units_per_lstm))
        U_Hat = self.intra_fc(U)
        U_Hat = tf.reshape(U_Hat, (self.batch_size, self.num_overlapping_chunks*self.chunk_size*self.num_filters_in_encoder))
        LN_U_Hat = self.intra_ln(U_Hat)
        LN_U_Hat = tf.reshape(LN_U_Hat, (self.batch_size, self.num_overlapping_chunks, self.chunk_size, self.num_filters_in_encoder))
        T_Hat = T + LN_U_Hat

        # Inter-Chunk Processing
        T_Hat = tf.transpose(T_Hat, [0, 2, 1, 3])
        T_Hat_shaped = tf.reshape(T_Hat, (self.batch_size*self.chunk_size, self.num_overlapping_chunks, self.num_filters_in_encoder))
        V = self.inter_rnn(T_Hat_shaped)
        V = tf.reshape(V, (self.batch_size*self.chunk_size*self.num_overlapping_chunks, self.units_per_lstm))
        V_Hat = self.inter_fc(V)
        V_Hat = tf.reshape(V_Hat, (self.batch_size, self.num_overlapping_chunks*self.fc_units*self.chunk_size))
        LN_V_Hat = self.inter_ln(V_Hat)
        T_Out = tf.reshape(LN_V_Hat, (self.batch_size, self.chunk_size, self.num_overlapping_chunks, self.fc_units))
        if not self.is_last_dprnn:
            T_Out = T_Hat + T_Out
        T_Out = tf.transpose(T_Out, [0, 2, 1, 3])

        return T_Out

class TasnetWithDprnn():
    def __init__(self, batch_size, model_weights_file, num_filters_in_encoder,
                 encoder_filter_length, chunk_size, num_full_chunks, units_per_lstm,
                 num_dprnn_blocks, samplerate_hz):
        self.batch_size = batch_size
        self.model_weights_file = model_weights_file
        self.num_dprnn_blocks = num_dprnn_blocks
        self.encoder_filter_length = encoder_filter_length
        self.num_filters_in_encoder = num_filters_in_encoder
        self.encoder_hop_size = encoder_filter_length // 2
        self.num_full_chunks = num_full_chunks
        self.signal_length_samples = chunk_size*num_full_chunks
        self.chunk_size = chunk_size
        self.chunk_advance = chunk_size // 2
        self.units_per_lstm = units_per_lstm
        self.num_overlapping_chunks = num_full_chunks*2-1
        self.num_speakers = 2
        self.samplerate_hz = samplerate_hz
        self.model = self.generate_model()

    def segment_encoded_signal(self, x):
        x1 = tf.reshape(x, (self.batch_size, self.signal_length_samples//self.chunk_size, self.chunk_size, self.num_filters_in_encoder))
        x2 = tf.roll(x, shift=-self.chunk_advance, axis=1)
        x2 = tf.reshape(x2, (self.batch_size, self.signal_length_samples//self.chunk_size, self.chunk_size, self.num_filters_in_encoder))
        x2 = x2[:, :-1, :, :] # Discard last segment with invalid data

        x_concat = tf.concat([x1, x2], axis=1)
        x = x_concat[:, ::self.num_full_chunks, :, :]
        for i in range(1, self.num_full_chunks):
            x = tf.concat([x, x_concat[:, i::self.num_full_chunks, :, :]], axis=1)
        return x

    def overlap_and_add_mask_segments(self, x):
        x = tf.transpose(x, [0, 3, 1, 2])
        x = tf.signal.overlap_and_add(x, self.chunk_advance)
        return tf.transpose(x, [0, 2, 1])

    def overlap_and_add_in_decoder(self, x):
        return tf.signal.overlap_and_add(x, self.encoder_hop_size)

    def generate_model(self):
        # Input
        network_input = keras.Input(shape=[self.signal_length_samples])
        # Encoder
        encoded = keras.layers.Conv1D(filters=self.num_filters_in_encoder, \
                                      kernel_size=self.encoder_filter_length,  \
                                      strides=self.encoder_hop_size, use_bias=False, \
                                      padding="same")(tf.expand_dims(network_input, axis=2))
        # Segmentation
        dprnn_in_out = keras.layers.Lambda(self.segment_encoded_signal)(encoded)
        # Dual-Path RNN blocks
        for b in range(1, self.num_dprnn_blocks+1):
            dprnn_in_out = DprnnBlock(1, is_last_dprnn=(b == self.num_dprnn_blocks), tasnet_with_dprnn=self)(dprnn_in_out)
        # Overlap + add mask segments
        masks = keras.layers.Lambda(self.overlap_and_add_mask_segments)(dprnn_in_out)
        # Apply speaker masks to encoded mixture signal
        encoded_spk0 = encoded*masks[:, :, :self.num_filters_in_encoder]
        encoded_spk1 = encoded*masks[:, :, self.num_filters_in_encoder:]
        # Decode speaker0
        decoded_spk0 = keras.layers.Dense(units=self.encoder_filter_length, use_bias=False)(encoded_spk0)
        decoded_spk0 = keras.layers.Lambda(self.overlap_and_add_in_decoder)(decoded_spk0)[:, :self.signal_length_samples]
        # Decode speaker1
        decoded_spk1 = keras.layers.Dense(units=self.encoder_filter_length, use_bias=False)(encoded_spk1)
        decoded_spk1 = keras.layers.Lambda(self.overlap_and_add_in_decoder)(decoded_spk1)[:, :self.signal_length_samples]
        # Stack decoded speaker signals for single tensor output
        decoded = tf.stack([decoded_spk0, decoded_spk1], axis=1)
        # Generate model
        model = keras.Model(inputs=network_input, outputs=decoded)
        if not self.model_weights_file == None:
            model.load_weights(self.model_weights_file)
            print('Loaded weights from', self.model_weights_file)
        return model

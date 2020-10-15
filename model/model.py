from tensorflow.keras.layers import Conv1D, LSTM, Bidirectional, LSTMCell
from modules import n_layer_1d_convolution, postnet
from tensorflow.keras import Model
from zoneout import ZoneoutWrapper
from attention import LocationSensitiveAttenetion
from wrappers import DecoderPrenetWrapper, ConcatOutputAndAttentionWrapper, OutputProjectionWrapper
import tensorflow as tf
import tensorflow_addons as tfa
from helpers import CustomTrainingHelper, CustomTestHelper

VOCABULARY_SIZE=100

def create_model(data, config, is_training=True):

    with tf.compat.v1.variable_scope("tacotron2", reuse=tf.AUTO_REUSE):

        inputs, input_sequences_length,target, target_sequences_length, \
            target_inputs = data['inputs'],data['input_sequences_length'],data['target'], \
                            data['target_sequences_length'],data['target_inputs']


        batch_size = tf.shape(inputs)[0]

        # Start of Encoder layers
        embedding_table = tf.compat.v1.get_variable(name='embedding_table', shape=[VOCABULARY_SIZE, 128], dtype=tf.float32,
                                        initializer = tf.truncated_normal_initializer(stddev=0.5))

        embedding_layer = tf.nn.embedding_lookup(embedding_table, inputs)
        conv_outputs = n_layer_1d_convolution(embedding_layer, n=3, filter_width=5, channels=512, name="convolution_encoder")
        cell_fw = ZoneoutWrapper(cell = LSTMCell(256, name="encoder_lstm_forward"), zoneout_drop_prob=0.1, is_training=is_training)
        cell_bw = ZoneoutWrapper(cell = LSTMCell(256, name="encoder_lstm_backward"), zoneout_drop_prob=0.1, is_training=is_training)

        outputs, _ = tf.compat.v1.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            conv_outputs,
            sequence_length=input_sequence_lengths,
            dtype=tf.float32,
        )
        outputs = tf.concat(outputs, axis=1, name='concat_blstm_outputs')


        decoder_cell_layer_1 = ZoneoutWrapper(cell = LSTMCell(256, name='decoder_lstm_layer_1'), zoneout_drop_prob=0.1, is_training=is_training)
        decoder_cell_layer_2 = ZoneoutWrapper(cell = LSTMCell(256, name='decoder_lstm_layer_2'), zoneout_drop_prob=0.1, is_training=is_training)

        attention_mechanism = LocationSensitiveAttention(num_units=128, memory=outputs, name='decoder_attention_mechanism')


        decoder_cell_layer_1 = DecoderPrenetWrapper(decoder_cell_layer_1)

        decoder_cell_layer_1 = tfa.seq2seq.AttentionWrapper(cell=decoder_cell_layer_1,
                                                            attention_mechanism=attention_mechanism,
                                                            output_attention=False,
                                                            alignment_history=True)

        decoder_cell_layer_1 = ConcatOutputAndAttentionWrapper(decoder_cell_layer_1)

        decoder_cell_layer_2 = OutputProjectionWrapper(decoder_cell_layer_2,
                                                        linear_projection_size=config['data']['num_mel_bins'])

        stacked_decoder_cell = tf.compat.v1.nn.rnn_cell.MultiRNNCell([decoder_cell_layer_1, decoder_cell_layer_2])

        if is_training:
            helper = CustomTrainingHelper(inputs=target_inputs,
                                        sequence_length=target_sequence_lengths,
                                        target_inputs=target_inputs,
                                        batch_size=batch_size,
                                        teacher_forcing_p=1.0)

        else:
            helper = CustomTestHelper(inputs=target_inputs,
                                        sequence_length=target_sequence_lengths,
                                        target_inputs=target_inputs,
                                        batch_size=batch_size,
                                        teacher_forcing_p=1.0)


        decoder = tfa.seq2seq.BasicDecoder(cell=stacked_decoder_cell,
                                            helper=helper,
                                            initial_state=stacked_decoder_cell.zero_state(batch_size, tf.float32))

        (mel_outputs, _), final_decoder_state, _ = tfa.seq2seq.dynamic_decode(decoder)

        residual_mels = postnet(mel_outputs)

        residual_mels = tf.compat.v1.reduce_mean(residual_mels, axis=-1)

        return mel_outputs, residual_added_mels

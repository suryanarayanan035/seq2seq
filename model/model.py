from tensorflow.keras.layers import Conv1D, LSTM, Bidirectional, LSTMCell
from modules import n_layer_1d_convolution
from tensorflow.keras import Model
from zoneout import ZoneoutWrapper
from attention import LocationSensitiveAttenetion
from wrappers import DecoderPrenetWrapper

VOCABULARY_SIZE=100

def create_model(data,config,is_training=True):

    with tf.varaiable_scope("tacotron2",reuse=tf.AUTO_REUSE):

        inputs,input_sequences_length,target,target_sequences_length, \
            target_inputs = data['inputs'],data['input_sequences_length'],data['target'], \
                            data['target_sequences_length'],data['target_inputs']


        batch_size=tf.shape(inputs)[0]

        # Start of Encoder layers
        embedding_table = tf.Variable(name='embedding_table',shape=[VOCABULARY_SIZE,128],dtype=tf.float32,
                                        initializer = tf.truncated_normal_initializer(stddev=0.5))

        embedding_layer = tf.nn.embedding_lookup(embedding_table,inputs)
        conv_outputs = n_layer_1d_convolution(embedding_layer,n=3,filter_width=5,channels=512,name="convolution_encoder")
        cell_fw = ZoneoutWrapper(LSTMCell(256,name="encoder_lstm_forward"),0.1,is_training=is_training)
        cell_bw = ZoneoutWrapper(LSTMCell(256,name="encoder_lstm_backward"),0.1,is_training=is_training)

        outputs = tf.keras.layers.Bidirectional(cell_fw,backward_layer=cell_bw)

        outputs = tf.concat(outputs, axis=1, name='concat_blstm_outputs')

        decoder_cell_layer_1 = ZoneoutWrapper(LSTMCell(256, name='decoder_lstm_layer_1'), zoneout_drop_prob=0.1, is_training=is_training)
        decoder_cell_layer_2 = ZoneoutWrapper(LSTMCell(256, name='decoder_lstm_layer_2'), zoneout_drop_prob=0.1, is_training=is_training)

        attention_mechanism = LocationSensitiveAttenetion(num_units=128,name="decoder_attention_mechanism")

        decoder_cell_layer_1 = DecoderPrenetWrapper(decoder_cell_layer_1)
        decoder_cell_layer_1 = tfa.seq2seqAttentionWrapper()
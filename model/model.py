from tensorflow.keras.layers import Conv1D,LSTM,Bidirectional
from modules import n_layer_1d_convolution
from tensorflow.keras import Model
from zoneout import ZoneoutWrapper

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
        
        
        cell_fw = ZoneoutWrapper(tf.keras.layers.LSTMCell(256,name="encoder_lstm_forward"),0.1,is_training=is_training)
        cell_bw = ZoneoutWrapper(tf.keras.layers.LSTMCell(256,name="encoder_lstm_backwarf),0.1,is_training=is_training)
        
        outputs = tf.keras.layers.Bidirectional(cell_fw,backward_layer=cell_bw)
        
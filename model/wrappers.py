import tensorflow as tf
from tensorflow.keras.layers import SimpleRNNCell
from zoneout import ZoneoutWrapper
from modules import decoder_prenet

class DecoderPrenetWrapper(SimpleRNNCell):
    def __init__(self,cell,is_training=True):
        super(DecoderPrenetWrapper,self).__init__(256)
        self._cell = cell
        self.is_training=is_training

    @property
    def _state_size(self):
        return self._state_size

    @property
    def _output_size(self):
        return self._cell.state_size
    
    @property
    def __call__(self,inputs,state):
        prenet_out = decoder_prenet(inputs,self.is_training)
        return self._cell(prenet_out,state)

    def zero_state(self,batch_size,dtype):
        return self._cell.zero_state(batch_size,dtype)

cell= ZoneoutWrapper(cell=tf.keras.layers.LSTMCell(256, name='decoder_lstm_layer_1'), zoneout_prob=0.1,
                                      is_training=False)
cell = DecoderPrenetWrapper(cell)
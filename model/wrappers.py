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


class ConcatOutputAndAttentionWrapper(SimpleRNNCell):
    def __init__(self,cell):
        super(ConcatOutputAndAttentionWrapper,self).__init__()
        self._cell=_cell

    @property
    def state_size(self):
        return self._cell.state_size 

    @property
    def _output_size(self):
        return self._cell.state_size + self._cell.state_size.attention 

    def call(self,inputs,state):
        output,res_state = self._cell(inputs,state)
        return tf.concat([output,res_state.attention],axis=1),res_state
    
    def zero_state(self,batch_size,dtype):
        return self._Cell.zero_state(batch_size,dtype)

class OutputProjectionWrapper(SimpleRNNCell):
    def __init__(self,cell,linear_projection_size):
        super(OutputProjectionWrapper,self).__init__()
        self._cell = cell
        self.linear_projection_size=linear_projection_size
        self.out_projection_layer = tf.layers.Dense(units=linear_projection_size,activation="relu",name="output_projection")

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self.linear_projection_size

    def call(self,inputs,state):
        output,res_stae = self._cell(inputs,state)
        out_projection = self.out_projection_layer(output)

        return out_projection,res_state

    def zero_state(Self,batch_size,dtype):
        return self._cell.zero_state(batch_size,dtype)
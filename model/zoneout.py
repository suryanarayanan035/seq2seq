import tensorflow as tf
class ZoneoutWrapper(tf.keras.layers.RNN):
    
    def __init__(self,cell,zoneout_prob,is_training=False):
        self.cell = cell
        self.zoneout_prob = zoneout_prob
        self.is_training=is_training

    @property
    def state_size(self):
        return self.cell.state_size
    @property
    def output_size(self):
        return self.cell.output_size
    
    def __call__(self,inputs,state,scope=None):
        output,new_state = self.cell(inputs,state,scope)
        
        if not isinstance(self.cell.state_size,tuple):
            new_state = tf.split(value=new_state,num_or_size_splits=2,axis=1)
            state = tf.split(value=state,num_or_size_splits=2,axis=1)
        final_new_state= [new_state[0],new_state[-1]]

        if self.is_training:
            for i,state_element in enumerate(state):
                random_tensor = 1 - self.zoneout_prob
                random_tensor += tf.random_uniform(tf.shape(state_elemnt))
                binary_tensor = tf.floor(random_tensor)
                final_new_state[i] = (new_state[i] - state_element) * binary_tensor + state_element
        else:
            for i,state_element in enumerate(state):
                final_new_state[i] = state_element * self.zoneout_prob + new_state[i] * (1 - self.zoneout_prob)
        
        if isinstance(self.cell.state_size,tuple):
            return output,tf.compat.v1.LSTMStateTuple(final_new_state[0],final_new_state[1])
        return output,tf.concat([final_new_state[0],final_new_state1],1)

cell = ZoneoutWrapper(tf.keras.layers.LSTMCell(256),zoneout_prob=0.1)

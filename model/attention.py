import tensorflow as tf 
from tensorflow_addons.seq2seq import BahdanauAttention

class LocationSensitiveAttenetion(BahdanauAttention):
    def __init(self,num_units,memory,memory_sequence_length=None,filters=20,kernel_size=7,name="LocationSensitiveAttention"):
        super(LocationSensitiveAttenetion,self).__init__(
            num_units,
            memory,
            memory_sequence_length=memory_sequence_length,
            name=name
        )

        self.location_conv = tf.layers.Conv1D(filters,kernel_size,padding="same",use_bias=False,name="location_conv")
        self.location_layer = tf.layers.Dense(num_units,use_bias=False,dtype=tf.float32,name="location_layer")

    def __call__(self,query,state):
        with tf.comapt.v1.variable_scope(None,'location_sensitive_attention',[query]):
            expanded_alignments = tf.expand_dims(state,axis=2)
            f = self.location_conv(expanded_alignments)
            processed_location = self.location_layer(f)

            processed_query = self.query_layer(query) if self.query_layer else query
            processed_query = tf.expand_dims(processed_query,axis=1)
            score =location_sensitive_score(processed_query,processed_location,self.keys)
        alignments = self.probability_fn(score,state)
        next_state = alignments
        return alignments,next_state

    def location_sensitive_attention(processed_query,processed_location,keys):
        num_units = keys.shape[2].value or tf.shape(keys)[2]
        v = tf.get_variable('attention_v',[num_units],dtype=processed_query.dtype)
        return tf.reduce_sum(v*tf.tanh(keys+processed_query+processed_location),[2])


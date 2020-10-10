import random
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
from tensorflow_addons.seq2seq import Sampler

class CustomTrainingHelper(Sampler):
    def __init__(self, inputs, sequence_length, target_inputs, batch_size, time_major=False, name=None, teacher_forcing_p=0.5):
        self._lengths = sequence_length
        self._teacher_forcing_p = teacher_forcing_p
        self._batch_size = batch_size
        self._output_dim = inputs.shape[2]
        self._inputs = inputs
        self._target_inputs = target_inputs
        self.stop_token_projection_layer = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid,
                                                           name='stop_token_projection')

    @property
    def inputs(self):
        return self._inputs

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def batch_size(self):
        return self._batch_size

    def initialize(self, name=None):
        finished = tf.tile([False], multiples=[self._batch_size])
        next_inputs = tf.tile([[0.0]], [self._batch_size, self._output_dim])
        return finished, next_inputs

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        finished = tf.greater_equal(time+1, self._lengths)
        if random.random() > self._teacher_forcing_p:
            # Again, note that outputs are [batch_size, output_dimension]
            next_inputs = outputs
        else:
            # target_inputs were passed by us and had the time axis too
            next_inputs = self._target_inputs[:, time, :]

        return finished, next_inputs, state

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

    def sample(self, time, outputs, state, name=None):
        # We are not using this but have to return a tensor
        return tf.tile([0], multiples=[self._batch_size])


class CustomTestHelper(Sampler):
    def __init__(self, inputs, sequence_length, target_inputs, batch_size, time_major=False, name=None):
        self._lengths = sequence_length
        self._batch_size = batch_size # if not time_major else inputs.shape[1]
        self._output_dim = inputs.shape[2]
        self._inputs = inputs
        self._target_inputs = target_inputs
        self.stop_token_projection_layer = tf.keras.layers.Dense(units=1, activation=tf.nn.sigmoid,
                                                           name='stop_token_projection')

    @property
    def inputs(self):
        return self._inputs

    @property
    def sequence_length(self):
        return self._sequence_length

    @property
    def batch_size(self):
        return self._batch_size

    def initialize(self, name=None):
        finished = tf.tile([False], multiples=[self._batch_size])
        next_inputs = tf.tile([[0.0]], [self._batch_size, self._output_dim])
        return finished, next_inputs

    def next_inputs(self, time, outputs, state, name=None, **unused_kwargs):
        finished = tf.reduce_all(input_tensor=tf.equal(outputs, 0.0), axis=1)
        next_inputs = outputs
        return finished, next_inputs, state

    @property
    def sample_ids_shape(self):
        return tf.TensorShape([])

    @property
    def sample_ids_dtype(self):
        return tf.int32

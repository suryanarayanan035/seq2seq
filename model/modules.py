import tensorflow as tf

def n_layer_1d_convolution(inputs, n, filter_width, channels, name="1d_convolution", is_training=False):
    conv_output = inputs

    for _ in range(n):
        conv_output = tf.keras.layers.Conv1D(conv_output, filters=channels, kernel_size=filter_width,
                                      padding="same", name=name+'_layer'+str(_))
        conv_output = tf.keras.layers.Dropout(conv_output, rate=0.5, name=name+'_layer'+str(_)+"_dropout", training=is_training)

    return conv_output

def decoder_prenet(inputs, is_training=False, name="decoder_prenet"):
    output = tf.keras.layer.Dense(256, activation="relu", name=name+"_layer_1")(inputs)
    if is_training:
        output = tf.keras.layers.Dropout(0.5, name=name+"_droput_layer1")(output)
    output= tf.keras.layers.Dense(256, activation="relu", name=name+"_layer_2")(output)
    if is_training:
         output = tf.keras.layers.Dropout(0.5, name=name+"_droput_layer1")(output)

    return output

def postnet(inputs, is_training=False, name="postnet"):
    conv_output = tf.expand_dims(inputs, -1)
    for _ in range(4):
        conv_output = tf.keras.layers.Conv2D(filers=512, kernel_size=(1,5),
                      padding="same", name=name+"_layer"+str(_), activation="tanh")(conv_output)
        conv_output = tf.keras.layers.Dropout(0.5, name=name + '_layer' + str(_) + "_dropout", training=is_training)(conv_output)

    return conv_output

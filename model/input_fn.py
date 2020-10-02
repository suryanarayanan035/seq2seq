import tensorflow as tf 

def parse_csv_line(line,vocabulary,config):
    fields = tf.decode_csv(line,config['data']['csv_column_defaults'])
    features = dict(zip(config['data']['csv_columns'],fields))
    text = tf.compat.v1.string_split(features[config['data']['csv_columns'][0]],sep="")
    text_idx = tf.SparseTensor(text.indices,tf.map_fn(vocabulary.text2idx,text.values,dtype="tf.int64"))
    text_idx = tf.sparse_tensor_to_dense(text_idx)
    text_idx = tf.squeeze(text_idx)

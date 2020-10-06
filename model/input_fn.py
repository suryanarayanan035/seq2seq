import tensorflow as tf 
import librosa
def parse_csv_line(line,vocabulary,config):

    #Text file Loading
    fields = tf.io.decode_csv(line,config['data']['csv_column_defaults'])
    features = dict(zip(config['data']['csv_columns'],fields))
    text = tf.compat.v1.string_split(features[config['data']['csv_columns'][0]],sep="")
    text_idx = tf.SparseTensor(text.indices,tf.map_fn(vocabulary.text2idx,text.values,dtype="tf.float32"))
    text_idx = tf.sparse_tensor_to_dense(text_idx)
    text_idx = tf.squeeze(text_idx)

    #Audio filesloading
    audio_path,sample_rate = tf.read_file(features[config['data']['csv_columns'][1]])
    waveform =tf.audio.decode_wav(audio_path,desired_samples=config["data"]["sample_rate"])
    stfts = tf.signal.stft(waveform,frame_length = config["data"]["frame_length"],
                           frame_step=config["data"]["frame_step"],fft_length=config["data"]["fft_length"]) 
    magnitude_spectrograms = tf.abs(stfts)
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value

    lower_edge_hertz,upper_edge_hertz,num_mel_bins = config['data']['lower_edge_hertz'],config['data']['upper_edge_hertz'],config['data']['num_mel_bins']
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins,num_spectrogram_bins,config['data']['sample_rate'],lower_edge_hertz,upper_edge_hertz)
    mel_spectrograms = tf.tensordot(magnitude_spectrograms,linear_to_mel_weight_matrix,1)
    mel_spectrograms = tf.squeeze(mel_spectrograms)

    end_tensor = tf.tile([[0.0]],multiples=[1,tf.shape(mel_spectrograms)[-1]])
    targets = tf.concat([mel_spectrograms,end_tensor])

    start_tensor = tf.tile([[0.0]],multiples=[1,tf.shape(mel_spectrograms)[-1]])
    target_inputs = tf.concat([start_tensor,mel_spectrograms])

    target_sequence_lengths = tf.shape(targets)[0]

    return {
        "inputs":text_idx,
        "targets":targets,
        "input_sequences_length":input_sequences_length,
        "target_sequences_length":target_sequence_lengths,
        "target_inputs":target_inputs,
        "debug_data":waveform
    }

def train_input_fn(vocabulary,config):
    dataset = tf.data.TextLineDataset(config["general"]["input_csv"])
    dataset = dataset.map(lambda line:parse_csv_line(line,vocabulary,config))
    dataset = dataset.repeat(config["hyper_params"]["epochs"])
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.padded_batch(config['hyper_params']["batch_size"],
                padded_shapes={
                    "inputs":[None],
                    "targets":[None,config['data']['num_mel_bins']],
                    "input_sequences_length":[],
                    "target_sequences_length":[],
                    "target_inputs":[None,config['data']['num_mel_bins']],
                    "debug_data":[None,None]
                    
                })
    iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()



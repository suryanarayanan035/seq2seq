from collections import defaultdict
import tensorflow as tf 
from tensorflow.compat.v1 import py_func

class Vocabulary():
    def __init__(self):
        allowed_Characters = 'abcdefghijklmnopqrstuvwxyz1234567890!.,#$%@()=+*/'
        print("Allowed characters:")
        print(allowed_Characters)
        self.vocabulary_size=len(allowed_Characters)
        self.char2idx = dict([(allowed_Characters[i],i) for i in range(len(allowed_Characters))])
        print("char2idx characters:")
        print(self.char2idx)
        self.idx2char = dict([(value,key) for key,value in self.char2idx.items()])
        print("idx2char characters:")
        print(self.idx2char)
        self.char2idx = defaultdict(lambda: self.vocabulary_size,self.char2idx)
        self.idx2char = defaultdict(lambda:self.vocabulary_size,self.idx2char)
    
    def text2idx(self,text):
        encoded = tf.py_function(lambda x: self.char2idx[x.lower()],[text],tf.int64)
        return encoded
    

    
voc = Vocabulary()
voc.text2idx('suryanarayanan')
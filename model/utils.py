from collections import defaultdict
import tensorflow as tf 

class Vocabulary():
    def __init__(self):
        allowed_Characters = 'abcdefghijklmnopqrstuvwxyz1234567890!.,#$%@()=+*/'
        self.vocabulary_size=len(allowed_Characters)
        self.char2idx = dict([(allowed_Characters[i],i) for i in range(len(allowed_Characters))])
        self.idx2char = dict([(value,key) for key,value in self.char2idx.items()])
        self.char2idx = defaultdict(lambda: self.vocabulary_size,self.char2idx)
        self.idx2char = defaultdict(lambda:self.vocabulary_size,self.idx2char)
    
    def toLower(self,text):
        lower_case = tf.strings.lower(text)
        encoded = self.char2idx[lower_case.ref()]
        return encoded

    def text2idx(self,text):
        encoded = tf.py_function(func=self.toLower,inp=[text],Tout=tf.float32,)
        return encoded

    
    

